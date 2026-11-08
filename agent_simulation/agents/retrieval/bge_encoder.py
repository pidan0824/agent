"""Compact retrieval views and local Chinese BGE encoding without token truncation.

Views are indexes, not replacements for the source events. Long texts are split
into nonoverlapping token chunks. Normalized CLS vectors are averaged by body
token count and normalized again, preserving the existing event-level scorer."""

from functools import lru_cache, cached_property
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path

from agent_simulation.data.common import require, sha256
from agent_simulation import PROJECT_ROOT
from agent_simulation.inference import resolve_device

PROJECT = PROJECT_ROOT
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
RETRIEVAL_VERSION = "zh-views-v1:normalized-cls-token-weighted-chunks-batched-v2"


class LocalBGE:
    def __init__(self, path, *, cache_path=None, cache_max_bytes=2 * 1024**3,
                 cache_reserve_bytes=8 * 1024**3, device="auto"):
        self.path = Path(path).resolve()
        self._device_preference = device
        self._vectors = OrderedDict()
        self.cache_size = 8192
        self.batch_size = 8
        names = (
            "model.safetensors",
            "config.json",
            "vocab.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        )
        require(
            all((self.path / n).is_file() for n in names),
            f"Incomplete local Chinese BGE checkpoint: {self.path}",
        )
        hashes = {name: sha256(self.path / name) for name in names}
        digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        self.identity = f"bge-small-zh-v1.5:{digest}:{RETRIEVAL_VERSION}"
        from agent_simulation.agents.retrieval.vector_cache import VectorCache
        self.disk_cache = (VectorCache(cache_path, self.identity, max_bytes=cache_max_bytes,
                                       reserve_bytes=cache_reserve_bytes)
                           if cache_path is not None else None)

    @cached_property
    def device(self):
        return resolve_device(self._device_preference)

    @property
    def tokenizer(self):
        return self._tokenizer()

    @lru_cache(maxsize=1)
    def _tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(self.path), local_files_only=True)

    @lru_cache(maxsize=1)
    def model(self):
        from transformers import AutoModel

        model = AutoModel.from_pretrained(str(self.path), local_files_only=True).to(self.device)
        model.eval()
        model.requires_grad_(False)
        return model

    @cached_property
    def chunk_limit(self):
        config = json.loads((self.path / "config.json").read_text())
        return min(self.tokenizer.model_max_length, config["max_position_embeddings"])

    def chunks(self, text, *, query=False):
        require(type(text) is str and bool(text.strip()), "Empty embedding text")
        tokenizer = self.tokenizer
        body = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        prefix = tokenizer.encode(QUERY_INSTRUCTION, add_special_tokens=False) if query else []
        limit = self.chunk_limit
        budget = limit - tokenizer.num_special_tokens_to_add(pair=False) - len(prefix)
        require(budget > 0 and body, "No body token budget")
        parts = []
        for start in range(0, len(body), budget):
            segment = body[start : start + budget]
            encoded = tokenizer.prepare_for_model(
                prefix + segment,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=True,
            )
            require(len(encoded["input_ids"]) <= limit, "BGE chunk exceeds model limit")
            parts.append(
                {
                    "start": start,
                    "end": start + len(segment),
                    "body_ids": segment,
                    "encoded": encoded,
                }
            )
        return parts

    def encode_many(self, texts, *, query=False):
        """Batch missing text chunks; cache vectors only, never authorization or ranking."""
        import torch
        import numpy as np

        texts = list(texts)
        keys = [(query, text) for text in texts]
        result = {key: self._vectors[key] for key in set(keys) if key in self._vectors}
        missing = list(dict.fromkeys(key for key in keys if key not in result))
        if missing and self.disk_cache is not None:
            result.update(self.disk_cache.read(missing))
            missing = [key for key in missing if key not in result]
        # Bound temporary chunk storage even for large historical candidate pools.
        for start in range(0, len(missing), 64):
            group = missing[start:start + 64]
            chunks = [self.chunks(text, query=query) for _, text in group]
            flat = [(i, j, part) for i, parts in enumerate(chunks) for j, part in enumerate(parts)]
            flat.sort(key=lambda entry: len(entry[2]['encoded']['input_ids']))
            vectors = [[None] * len(parts) for parts in chunks]
            with torch.inference_mode():
                for offset in range(0, len(flat), self.batch_size):
                    batch_parts = flat[offset:offset + self.batch_size]
                    batch = self.tokenizer.pad([part['encoded'] for _, _, part in batch_parts],
                                               padding=True, return_tensors='pt').to(self.device)
                    encoded = torch.nn.functional.normalize(
                        self.model()(**batch).last_hidden_state[:, 0], p=2, dim=1).cpu()
                    for (i, j, _), vector in zip(batch_parts, encoded):
                        vectors[i][j] = vector
                for key, parts, values in zip(group, chunks, vectors):
                    weights = torch.tensor([len(p['body_ids']) for p in parts], dtype=torch.float32)
                    pooled = (torch.stack(values) * weights[:, None]).sum(dim=0) / weights.sum()
                    require(bool(torch.isfinite(pooled).all()) and pooled.norm().item() > 0,
                            'Invalid pooled BGE vector')
                    result[key] = torch.nn.functional.normalize(pooled, dim=0).numpy().copy()
        if self.disk_cache is not None:
            self.disk_cache.write({key: result[key] for key in missing})
        for key in keys:
            # Float32 is the encoder's original output, not a precision reduction.
            value = np.asarray(result[key], dtype=np.float32)
            value.setflags(write=False)
            result[key] = self._vectors[key] = value
            self._vectors.move_to_end(key)
            if len(self._vectors) > self.cache_size:
                self._vectors.popitem(last=False)
        return [result[key] for key in keys]

    def encode(self, text, query=False):
        return self.encode_many([text], query=query)[0]

    def __call__(self, text):
        return self.encode(text)

    def encode_query(self, text):
        return self.encode(text, True)


@lru_cache(maxsize=2)
def bge_encoder(path, cache_path=None, cache_max_bytes=2 * 1024**3, cache_reserve_bytes=8 * 1024**3,
                *, device="auto"):
    model = LocalBGE(path, cache_path=cache_path, cache_max_bytes=cache_max_bytes,
                     cache_reserve_bytes=cache_reserve_bytes, device=device)
    return model, model.identity


def default_encoder():
    return bge_encoder(
        os.environ.get("MEMORY_BGE_PATH", str(PROJECT / "models/bge-small-zh-v1.5")),
        os.environ.get("MEMORY_VECTOR_CACHE", str(PROJECT / "runs/retrieval_cache/vectors-v2.sqlite3")),
        int(float(os.environ.get("MEMORY_VECTOR_CACHE_GIB", "2")) * 1024**3),
        int(float(os.environ.get("MEMORY_VECTOR_RESERVE_GIB", "8")) * 1024**3),
        device=os.environ.get("MEMORY_BGE_DEVICE", "auto"),
    )


def encode_query(embedder, text):
    """Explicit test encoders can retain the simple callable contract."""
    method = getattr(embedder, "encode_query", embedder)
    return method(text)
