"""Extracts single-layer or multi-layer fused prefill activations from local open-weight models."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

if TYPE_CHECKING:
    from .base import TransformerActivationEncoder

logger = logging.getLogger(__name__)


class PrefillActivationExtractor:
    """Extracts hidden-state residual stream representations from local open-weight models.

    Supports both single-layer extraction (`layer_idx` / `target_layer`) and multi-layer
    residual stream fusion (`layer_indices` with `layer_fusion="concat"` or `"mean"`).
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM-135M",
        layer_idx: int = -1,
        pooling: Literal["last", "mean"] = "last",
        device: str = "cpu",
        local_files_only: bool = True,
        cache_file: str | None = ".cache/activations/smollm_activations.npz",
        encoder: TransformerActivationEncoder | None = None,
        target_layer: int | None = None,
        layer_indices: list[int] | None = None,
        layer_fusion: Literal["concat", "mean"] = "concat",
    ) -> None:
        if layer_fusion not in ("concat", "mean"):
            raise ValueError(f"layer_fusion must be 'concat' or 'mean', got '{layer_fusion}'.")

        self.model_name = model_name
        self.layer_idx = target_layer if target_layer is not None else layer_idx
        self.layer_indices = list(layer_indices) if layer_indices is not None else None
        self.layer_fusion: Literal["concat", "mean"] = layer_fusion
        self.pooling = pooling
        self.device = device
        self.local_files_only = local_files_only
        self.cache_file = Path(cache_file) if cache_file else None

        self._tokenizer: Any = None
        self._model: Any = None
        self._cache: dict[str, np.ndarray] = {}

        if encoder is not None:
            self._model = encoder.model
            self._tokenizer = encoder.tokenizer
            self.device = str(encoder.device)

        if self.cache_file and self.cache_file.exists():
            try:
                data = np.load(self.cache_file)
                self._cache = {k: data[k] for k in data.files}
                logger.info(f"Loaded {len(self._cache)} cached activations from {self.cache_file}")
            except Exception as e:
                logger.warning(f"Failed to load disk activation cache: {e}")

    @property
    def num_extracted_layers(self) -> int:
        """Returns number of transformer layers extracted per forward pass."""
        if self.layer_indices is not None and len(self.layer_indices) > 0:
            return len(self.layer_indices)
        return 1

    def _ensure_loaded(self) -> None:
        """Lazily loads the model and tokenizer onto the target device."""
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as err:
            raise ImportError(
                "The 'transformers' package is required for PrefillActivationExtractor. "
                "Install it via: pip install 'mechanistic-router[probe]'"
            ) from err

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
        if self.layer_indices is not None:
            key = (
                f"{prompt}|layers={self.layer_indices}"
                f"|fusion={self.layer_fusion}|pool={self.pooling}"
            )
            return hashlib.sha256(key.encode()).hexdigest()
        return hashlib.sha256(prompt.encode()).hexdigest()

    def _pool_hidden_state(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pools sequence token representations into a fixed-width batch matrix (B, D)."""
        if self.pooling == "last":
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_idx, seq_lengths]
        mask_expanded = attention_mask.unsqueeze(-1)
        return (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

    def extract_one(self, prompt: str) -> np.ndarray:
        """Extracts activation vector for a single prompt (shape: (feature_dim,))."""
        prompt_hash = self._hash_prompt(prompt)
        if prompt_hash in self._cache:
            return self._cache[prompt_hash]

        batch_activations = self.extract_batch([prompt])
        return np.asarray(batch_activations[0])

    def extract_batch(self, prompts: list[str], batch_size: int = 16) -> np.ndarray:
        """Extracts activation vectors for a batch of prompts (shape: (N, feature_dim))."""
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

        active_layers = (
            self.layer_indices
            if (self.layer_indices is not None and len(self.layer_indices) > 0)
            else [self.layer_idx]
        )

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
                attention_mask = inputs["attention_mask"]

                pooled_layers: list[torch.Tensor] = [
                    self._pool_hidden_state(outputs.hidden_states[idx], attention_mask)
                    for idx in active_layers
                ]

                if len(pooled_layers) == 1:
                    fused = pooled_layers[0]
                elif self.layer_fusion == "concat":
                    fused = torch.cat(pooled_layers, dim=-1)
                else:
                    fused = torch.stack(pooled_layers, dim=0).mean(dim=0)

                pooled_np = fused.float().cpu().numpy().astype(np.float32)

            for local_i, global_idx in enumerate(chunk_indices):
                vec = pooled_np[local_i]
                results[global_idx] = vec
                self._cache[self._hash_prompt(chunk_texts[local_i])] = vec

        if self.cache_file and prompts_to_compute:
            try:
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(str(self.cache_file), **self._cache)  # type: ignore[arg-type]
            except Exception as e:
                logger.warning(f"Failed to persist activation cache: {e}")

        return np.stack(results, axis=0)


__all__ = ["PrefillActivationExtractor"]
