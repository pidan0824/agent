"""Apply official ranking models to the current query and visible candidates."""

from copy import copy

import numpy as np

from agent_simulation.data.common import require


class OfficialRanking:
    def __init__(
        self, model, processor, text_model, tokenizer, item_id_map, user_id_map, *, device="cpu"
    ):
        self.model, self.processor = model, processor
        self.text_model, self.tokenizer, self.device = text_model, tokenizer, device
        self.item_id_map, self.user_id_map = item_id_map, user_id_map

    def features(self, context, ids):
        """Reuse the official collator with a fresh in-memory query-vector slot."""
        import torch

        user = context["user_id"]
        require(user in self.user_id_map, "User absent from official feature ID map")
        mapped_user = self.user_id_map[user]
        history = context["recently_clicked_item_ids"][-self.processor.max_history_len :]
        needed = set(ids) | set(history)
        require(needed <= self.item_id_map.keys(), "Product absent from official feature ID map")
        require(
            mapped_user in self.processor.user_features,
            "User absent from official profile features",
        )
        require(
            {self.item_id_map[i] for i in needed} <= self.processor.item_corpus.keys(),
            "Product absent from official corpus features",
        )
        batch = self.tokenizer(
            [context["query"]], max_length=32, truncation=True, padding=True, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            vector = self.text_model(**batch).last_hidden_state[:, 0, :].cpu().numpy()
        # The official preprocessing stores float16 before collate converts to float32.
        processor = copy(self.processor)
        processor.query_emb = vector.astype(np.float16)
        processor.session_id2idx = {str(context["request_id"]): 0}
        mapped_history = [self.item_id_map[i] for i in history]
        samples = [
            {
                "session_id": context["request_id"],
                "user_id": mapped_user,
                "target_item_id": self.item_id_map[item],
                "recently_clicked_item_ids": mapped_history,
            }
            for item in ids
        ]
        query_features, user_features, item_features, _ = processor.collate_fn(samples)
        return [
            {k: v.to(self.device) for k, v in features.items()}
            for features in (query_features, user_features, item_features)
        ]

    def __call__(self, context, ids):
        import torch

        features = self.features(context, ids)
        with torch.inference_mode():
            return torch.sigmoid(self.model(*features)).cpu().tolist()
