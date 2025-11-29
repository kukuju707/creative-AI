import os
from typing import List, Set
import numpy as np
from openai import OpenAI

from SeedCore.Config.ConfigInstance import ConfigSubsystem


class SeedPruner:
    """Prunes similar seeds based on cosine similarity using embedding vectors."""

    def __init__(self):

        self.client = OpenAI(api_key=ConfigSubsystem.get_config("config/excluded/credentials.json").get('OPENAI_API_KEY'))
        self._embedding_cache = {}

    def _get_config(self):
        return ConfigSubsystem.get_config("config/Pipeline/P_MultiStageGenerator.json")

    def _get_embedding_from_text(self, text: str) -> np.ndarray:

        # Check cache first
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        # Generate embedding using OpenAI API
        pruner_config = self._get_config().get("seed_pruner")
        embedding_model_name = pruner_config.get("embedding_model_name")
        embedding_response = self.client.embeddings.create(
            model=embedding_model_name,
            input=text
        )
        embedding = np.array(embedding_response.data[0].embedding)

        # Cache the embedding
        self._embedding_cache[text] = embedding

        return embedding

    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def prune_seeds(self, seeds: List[str], similarity_threshold) -> List[str]:

        if len(seeds) <= 1:
            return seeds

        embeddings = [self._get_embedding_from_text(seed) for seed in seeds]

        kept_indices: Set[int] = set()

        for i in range(len(seeds)):
            if i in kept_indices:
                continue

            is_similar = False
            for kept_idx in kept_indices:
                similarity = self._calculate_cosine_similarity(embeddings[i], embeddings[kept_idx])
                if similarity >= similarity_threshold:
                    is_similar = True
                    break

            if not is_similar:
                kept_indices.add(i)

        return [seeds[i] for i in sorted(kept_indices)]

    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()
