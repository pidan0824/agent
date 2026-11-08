"""Retrieve visible products with DPR."""

import numpy as np

from agent_simulation.data.common import require


class DPRRecall:
    def __init__(self, encoder, tokenizer, embeddings, item_ids, *, device="cpu"):
        self.encoder, self.tokenizer, self.device = encoder, tokenizer, device
        self.embeddings = embeddings
        self.item_index = {int(item): i for i, item in enumerate(item_ids)}
        require(
            len(self.item_index) == len(item_ids) == len(embeddings), "Recall index ID mismatch"
        )

    def __call__(self, context, count):
        import torch

        ids = context["market_item_ids"]
        try:
            indices = [self.item_index[item] for item in ids]
        except KeyError as error:
            raise ValueError("Recall index lacks visible market product " + str(error)) from error
        # Match official recall/dpr/test.py: apostrophe normalization, length 64,
        # unnormalized CLS vectors and inner-product retrieval.
        query = context["query"].replace("’", "'")
        batch = self.tokenizer(
            [query], max_length=64, truncation=True, padding="max_length", return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            vector = self.encoder(**batch).last_hidden_state[0, 0].float()
            if vector.device.type == "cpu":
                scores = np.asarray(self.embeddings[indices], dtype=np.float32) @ vector.numpy()
            else:
                # Keep the disk-backed catalogue on the host and bound each GPU transfer.
                block_size = 8192
                scores = np.empty(len(indices), dtype=np.float32)
                for start in range(0, len(indices), block_size):
                    end = start + block_size
                    rows = np.asarray(self.embeddings[indices[start:end]], dtype=np.float32)
                    block = torch.from_numpy(rows).to(vector.device)
                    scores[start:end] = (block @ vector).cpu().numpy()
        order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))[:count]
        return [{"item_id": ids[i], "recall_score": float(scores[i])} for i in order]
