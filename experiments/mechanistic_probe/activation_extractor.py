"""Extracts prefill activations from local open-weight models."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class PrefillActivationExtractor:
    """Extracts hidden-state residual stream representations from local open-weight models."""

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM-135M",
        layer_idx: int = -1,
        pooling: Literal["last", "mean"] = "last",
        device: str = "cpu",
        local_files_only: bool = True,
        cache_file: str | None = ".cache/activations/smollm_activations.npz",
    ) -> None:
        self.model_name = model_name
        self.layer_idx = layer_idx
        self.pooling = pooling
        self.device = device
        self.local_files_only = local_files_only
        self.cache_file = Path(cache_file) if cache_file else None

        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModel | None = None
        self._cache: dict[str, np.ndarray] = {}

        if self.cache_file and self.cache_file.exists():
            try:
                data = np.load(self.cache_file)
                self._cache = {k: data[k] for k in data.files}
                logger.info(f"Loaded {len(self._cache)} cached activations from {self.cache_file}")
            except Exception as e:
                logger.warning(f"Failed to load disk activation cache: {e}")

    def _ensure_loaded(self) -> None:
        """Lazily loads the model and tokenizer onto the target device."""
        if self._model is not None and self._tokenizer is not None:
            return

        logger.info(f"Loading local open-weight model: {self.model_name} onto {self.device}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModel.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        )
        self._model.to(self.device)
        self._model.eval()

    def _hash_prompt(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def extract_one(self, prompt: str) -> np.ndarray:
        """Extracts activation vector for a single prompt (shape: (hidden_dim,))."""
        prompt_hash = self._hash_prompt(prompt)
        if prompt_hash in self._cache:
            return self._cache[prompt_hash]

        batch_activations = self.extract_batch([prompt])
        return batch_activations[0]

    def extract_batch(self, prompts: list[str], batch_size: int = 16) -> np.ndarray:
        """Extracts activation vectors for a batch of prompts (shape: (N, hidden_dim))."""
        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None

        results: list[np.ndarray] = [np.array([])] * len(prompts)
        prompts_to_compute: list[tuple[int, str]] = []

        for idx, prompt in enumerate(prompts):
            p_hash = self._hash_prompt(prompt)
            if p_hash in self._cache:
                results[idx] = self._cache[p_hash]
            else:
                prompts_to_compute.append((idx, prompt))

        if not prompts_to_compute:
            return np.stack(results, axis=0)

        # Batch forward pass for uncached prompts
        for chunk_start in range(0, len(prompts_to_compute), batch_size):
            chunk = prompts_to_compute[chunk_start : chunk_start + batch_size]
            chunk_indices = [idx for idx, _ in chunk]
            chunk_texts = [p for _, p in chunk]

            inputs = self._tokenizer(
                chunk_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[self.layer_idx]  # (B, T, D)

                if self.pooling == "last":
                    # Last non-padding token
                    attention_mask = inputs["attention_mask"]
                    seq_lengths = attention_mask.sum(dim=1) - 1
                    batch_idx = torch.arange(hidden_states.size(0), device=self.device)
                    pooled = hidden_states[batch_idx, seq_lengths]
                else:
                    # Masked mean pooling
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)
                    pooled = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(
                        dim=1
                    ).clamp(min=1)

                pooled_np = pooled.float().cpu().numpy().astype(np.float32)

            for local_i, global_idx in enumerate(chunk_indices):
                vec = pooled_np[local_i]
                results[global_idx] = vec
                self._cache[self._hash_prompt(chunk_texts[local_i])] = vec

        if self.cache_file and prompts_to_compute:
            try:
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(self.cache_file, **self._cache)
            except Exception as e:
                logger.warning(f"Failed to persist activation cache: {e}")

        return np.stack(results, axis=0)
