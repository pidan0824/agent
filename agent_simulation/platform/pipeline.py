"""Compose recall, ranking and reranking for the configured platform."""

from copy import deepcopy
import math
import json
from pathlib import Path
import numpy as np
from agent_simulation.data.common import require


class RecommendationPipeline:
    def __init__(self, recall, ranking, *, identity, rerank=None, reranker_id="identity"):
        require(bool(identity) and bool(reranker_id), "Platform component identities are required")
        require(rerank is None or reranker_id != "identity", "Custom reranker requires an explicit ID")
        self.recall, self.ranking, self.rerank = recall, ranking, rerank
        self.identity, self.reranker_id = identity, reranker_id

    def search(self, context, candidate_size, top_k):
        require(
            type(top_k) is int and type(candidate_size) is int and 0 < top_k <= candidate_size,
            "Invalid platform candidate or Top-K size",
        )
        recalled = self.recall(deepcopy(context), candidate_size)
        ids = [r["item_id"] for r in recalled]
        candidates = set(ids)
        require(
            len(ids) <= candidate_size
            and len(ids) == len(candidates)
            and candidates <= set(context["market_item_ids"]),
            "Recall must return distinct visible market products",
        )
        ranking_ids = list(ids)
        scores = self.ranking(deepcopy(context), ranking_ids) if ids else []
        require(ranking_ids == ids, "Ranking must preserve candidate ID order")
        scores = list(map(float, scores))
        require(
            len(scores) == len(ids) and all(map(math.isfinite, scores)),
            "Ranking must score every recalled candidate",
        )
        ranked = [dict(row, ranking_score=score) for row, score in zip(recalled, scores)]
        ranked.sort(key=lambda row: (-row["ranking_score"], row["item_id"]))
        if self.rerank is None:
            selected = [row["item_id"] for row in ranked[:top_k]]
        else:
            selected = list(self.rerank(deepcopy(context), deepcopy(ranked), top_k))
            selected_ids = set(selected)
            require(
                len(selected) == min(top_k, len(ids))
                and len(selected_ids) == len(selected)
                and selected_ids <= candidates,
                "Re-ranking must return an ordered Top-K from candidates",
            )
        return selected, {
            "pipeline_id": self.identity,
            "reranker_id": self.reranker_id,
            "recall": recalled,
            "ranking": ranked,
            "reranked_item_ids": selected,
        }


def load_pipeline(manifest_path, *, rerank=None, reranker_id="identity"):
    """Load the models and product features listed in the local manifest."""
    from agent_simulation.inference import resolve_device
    from agent_simulation.platform.ranking import OfficialRanking
    from agent_simulation.platform.recall import DPRRecall

    path = Path(manifest_path).resolve()
    require(path.is_file(), "Missing KuaiSearch platform asset manifest: " + str(path))
    spec = json.loads(path.read_text())
    require(spec["schema"] == "kuaisearch-platform-assets-v1", "Unknown platform asset schema")
    required = (
        "recall_encoder",
        "recall_vectors",
        "recall_item_ids",
        "ranking_checkpoint",
        "ranking_text_encoder",
        "ranking_items",
        "ranking_users",
        "ranking_item_vectors",
        "ranking_embedding_index",
        "item_id_map",
        "user_id_map",
    )
    paths = {name: (path.parent / spec[name]).resolve() for name in required}
    for name, asset in paths.items():
        require(asset.exists(), f"Missing trained/prepared platform asset {name}: {asset}")
    import torch
    from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer

    device = resolve_device(spec.get("device", "auto"))
    recall_encoder = (
        BertModel.from_pretrained(
            paths["recall_encoder"], add_pooling_layer=False, local_files_only=True
        )
        .to(device)
        .eval()
    )
    recall_tokenizer = BertTokenizer.from_pretrained(paths["recall_encoder"], local_files_only=True)
    recall = DPRRecall(
        recall_encoder,
        recall_tokenizer,
        np.load(paths["recall_vectors"], mmap_mode="r"),
        np.load(paths["recall_item_ids"], allow_pickle=False),
        device=device,
    )
    from agent_simulation.platform.ranking import models
    from agent_simulation.platform.ranking.datasets import TrainingDataProcessor

    processor = TrainingDataProcessor.__new__(TrainingDataProcessor)

    def read_jsonl(asset, key):
        with asset.open() as stream:
            return {row[key]: row for line in stream if line.strip() for row in [json.loads(line)]}

    processor.item_corpus = read_jsonl(paths["ranking_items"], "item_id")
    processor.user_features = read_jsonl(paths["ranking_users"], "user_id")
    processor.max_history_len = spec["max_history_len"]
    require(
        type(processor.max_history_len) is int and processor.max_history_len > 0,
        "Invalid official Ranking history limit",
    )
    processor.item_emb = np.load(paths["ranking_item_vectors"], mmap_mode="r")
    processor.item_id2idx = {
        int(k): v for k, v in json.loads(paths["ranking_embedding_index"].read_text()).items()
    }
    model_name = spec["ranking_model"]
    names = {
        "DIN": "DINModel",
        "DNN": "DNNModel",
        "WideDeep": "WideDeepModel",
        "DCNv1": "DCNModel",
        "DCNv2": "DCNModel",
    }
    require(model_name in names, "Unknown official Ranking model")
    kwargs = dict(spec["ranking_model_kwargs"])
    if model_name.startswith("DCN"):
        kwargs["version"] = "v1" if model_name == "DCNv1" else "v2"
    with torch.device("meta"):
        model = getattr(models, names[model_name])(**kwargs)
    checkpoint = torch.load(paths["ranking_checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True, assign=True)
    model.to(device).eval().requires_grad_(False)
    text_model = (
        AutoModel.from_pretrained(paths["ranking_text_encoder"], local_files_only=True)
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(paths["ranking_text_encoder"], local_files_only=True)

    def mapping(name):
        return {int(k): int(v) for k, v in json.loads(paths[name].read_text()).items()}

    ranking = OfficialRanking(
        model,
        processor,
        text_model,
        tokenizer,
        mapping("item_id_map"),
        mapping("user_id_map"),
        device=device,
    )
    return RecommendationPipeline(
        recall, ranking, identity=spec["identity"], rerank=rerank, reranker_id=reranker_id,
    )
