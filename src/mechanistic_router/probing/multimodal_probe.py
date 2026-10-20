"""Multi-Modal Vision-Language Cross-Attention & Patch-Token Residual Probing.

Analyzes joint vision-patch and text-token residual stream representations
H = [H_vision; H_text] in R^{(N_v + N_t) x D} for Vision-Language Model (VLM)
tier routing and adaptive visual token compression:
1. Cross-Modal Attention & Alignment (Liu et al., NeurIPS 2023):
   Computes scaled dot-product cross-attention A_{t -> v} = softmax(H_t H_v^T / sqrt(D))
   measuring how strongly text queries attend to visual patch tokens.
2. FastV + PruMerge Visual Patch Pruning & Merging (Chen et al., ECCV 2024;
   Shang et al., 2024):
   Selects top-K high-attention visual patches and merges pruned peripheral patches
   into their nearest retained anchor patches via attention-weighted aggregation.
3. Visual Effective Dimensionality (d_eff^{vision}) & Multi-Modal Complexity:
   Combines SVD spectral rank of visual patches, cross-modal attention entropy,
   and linguistic probe probability into a calibrated VLM routing score.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .linear_probe import LinearActivationProbe


@dataclass(frozen=True)
class MultiModalProbeTelemetry:
    """Mechanistic telemetry for vision-language residual stream probing and patch pruning."""

    num_vision_patches: int
    num_text_tokens: int
    vision_effective_dim: float
    cross_modal_alignment: float
    cross_modal_entropy: float
    prob_complex: float
    retained_patch_indices: list[int]
    pruned_patch_indices: list[int]
    patch_retention_ratio: float
    prumerge_reconstruction_error: float

    def to_dict(self) -> dict[str, Any]:
        """Serializes multimodal probe telemetry to a JSON-compatible dictionary."""
        return asdict(self)


class MultiModalActivationProbe:
    """Probes vision-language residual activations and performs FastV/PruMerge patch reduction."""

    def __init__(
        self,
        patch_retention_ratio: float = 0.50,
        vision_weight: float = 0.45,
        text_weight: float = 0.55,
    ) -> None:
        if not (0.0 < patch_retention_ratio <= 1.0):
            raise ValueError(
                f"patch_retention_ratio must be in (0, 1], got {patch_retention_ratio}."
            )
        if vision_weight < 0.0 or text_weight < 0.0 or (vision_weight + text_weight) <= 0.0:
            raise ValueError(
                "vision_weight and text_weight must be non-negative with positive sum."
            )

        total_w = float(vision_weight + text_weight)
        self.patch_retention_ratio = float(patch_retention_ratio)
        self.vision_weight = float(vision_weight / total_w)
        self.text_weight = float(text_weight / total_w)

    @staticmethod
    def _validate_modalities(
        vision_patches: np.ndarray,
        text_tokens: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validates and normalizes vision_patches (N_v, D) and text_tokens (N_t, D)."""
        v_arr = np.asarray(vision_patches, dtype=np.float64)
        t_arr = np.asarray(text_tokens, dtype=np.float64)
        if v_arr.ndim == 1:
            v_arr = v_arr[None, :]
        if t_arr.ndim == 1:
            t_arr = t_arr[None, :]
        if v_arr.ndim != 2 or t_arr.ndim != 2:
            raise ValueError("vision_patches and text_tokens must be 1D or 2D arrays.")
        if v_arr.shape[0] == 0 or t_arr.shape[0] == 0:
            raise ValueError("vision_patches and text_tokens must be non-empty.")
        if v_arr.shape[1] != t_arr.shape[1] or v_arr.shape[1] == 0:
            raise ValueError(
                f"Hidden dimension mismatch: vision={v_arr.shape[1]}, text={t_arr.shape[1]}."
            )
        return v_arr, t_arr

    def compute_cross_modal_attention(
        self,
        vision_patches: np.ndarray,
        text_tokens: np.ndarray,
    ) -> np.ndarray:
        """Computes text-to-vision cross-attention matrix A_{t->v} of shape (N_t, N_v)."""
        v_arr, t_arr = self._validate_modalities(vision_patches, text_tokens)
        d_dim = float(v_arr.shape[1])
        logits = (t_arr @ v_arr.T) / math.sqrt(d_dim)
        logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        denom = np.maximum(np.sum(exp_logits, axis=-1, keepdims=True), 1e-12)
        return np.asarray(exp_logits / denom, dtype=np.float64)

    def prune_and_merge_patches(
        self,
        vision_patches: np.ndarray,
        text_tokens: np.ndarray,
        retention_ratio: float | None = None,
    ) -> tuple[np.ndarray, list[int], list[int], float]:
        """Applies FastV top-K patch selection and PruMerge similarity-weighted token merging."""
        v_arr, t_arr = self._validate_modalities(vision_patches, text_tokens)
        n_v = int(v_arr.shape[0])
        ratio = (
            float(np.clip(retention_ratio, 1e-3, 1.0))
            if retention_ratio is not None
            else self.patch_retention_ratio
        )

        cross_attn = self.compute_cross_modal_attention(v_arr, t_arr)
        patch_scores = np.mean(cross_attn, axis=0)  # (N_v,)
        patch_scores = patch_scores / max(float(np.sum(patch_scores)), 1e-12)

        k_keep = min(n_v, max(1, int(math.ceil(ratio * float(n_v)))))
        ranked = sorted(range(n_v), key=lambda idx: (float(patch_scores[idx]), -idx), reverse=True)
        retained_indices = sorted(ranked[:k_keep])
        retained_set = set(retained_indices)
        pruned_indices = [i for i in range(n_v) if i not in retained_set]

        if not pruned_indices:
            return v_arr.copy(), retained_indices, [], 0.0

        # Normalize vectors for cosine similarity matching between pruned and retained patches
        v_norms = np.maximum(np.linalg.norm(v_arr, axis=1, keepdims=True), 1e-12)
        v_unit = v_arr / v_norms

        retained_unit = v_unit[retained_indices]  # (K_v, D)
        merged_patches = v_arr[retained_indices].copy() * patch_scores[retained_indices, None]
        cluster_weights = patch_scores[retained_indices].copy()

        for p_idx in pruned_indices:
            sims = retained_unit @ v_unit[p_idx]
            best_local = int(np.argmax(sims))
            w_p = float(patch_scores[p_idx])
            merged_patches[best_local] += w_p * v_arr[p_idx]
            cluster_weights[best_local] += w_p

        safe_weights = np.maximum(cluster_weights[:, None], 1e-12)
        merged_normalized = merged_patches / safe_weights

        # Reconstruction error between original attention-pooled context and PruMerge context
        orig_context = np.sum(v_arr * patch_scores[:, None], axis=0)
        unmerged_top_context = np.sum(
            v_arr[retained_indices] * patch_scores[retained_indices, None], axis=0
        ) / max(float(np.sum(patch_scores[retained_indices])), 1e-12)
        denom_norm = max(float(np.linalg.norm(orig_context)), 1e-12)
        # Blend PruMerge cluster distortion (weighted variance within clusters)
        intra_cluster_distortion = 0.0
        for p_idx in pruned_indices:
            sims = retained_unit @ v_unit[p_idx]
            best_local = int(np.argmax(sims))
            anchor_idx = retained_indices[best_local]
            intra_cluster_distortion += float(patch_scores[p_idx]) * float(
                np.linalg.norm(v_arr[p_idx] - v_arr[anchor_idx])
            )
        recon_err = float(
            min(
                1.0,
                0.5 * (intra_cluster_distortion / denom_norm)
                + 0.5 * (float(np.linalg.norm(orig_context - unmerged_top_context)) / denom_norm),
            )
        )
        return merged_normalized, retained_indices, pruned_indices, round(recon_err, 6)

    def evaluate(
        self,
        vision_patches: np.ndarray,
        text_tokens: np.ndarray,
        text_probe: LinearActivationProbe | None = None,
        retention_ratio: float | None = None,
    ) -> MultiModalProbeTelemetry:
        """Evaluates multimodal residual telemetry, patch pruning, and routing complexity."""
        v_arr, t_arr = self._validate_modalities(vision_patches, text_tokens)
        n_v, d_dim = v_arr.shape
        n_t = int(t_arr.shape[0])

        # 1. Visual Effective Dimensionality d_eff^{vision}
        _, s_vals, _ = np.linalg.svd(v_arr, full_matrices=False)
        energy = s_vals**2
        total_energy = float(np.sum(energy))
        if total_energy > 1e-12 and energy.size > 1:
            p_spec = np.maximum(energy / total_energy, 1e-12)
            spec_ent = float(-np.sum(p_spec * np.log(p_spec)))
            d_eff_v = float(math.exp(spec_ent))
            norm_d_eff = float(
                np.clip((d_eff_v - 1.0) / max(1.0, float(min(n_v, d_dim) - 1)), 0.0, 1.0)
            )
        else:
            d_eff_v = 1.0
            norm_d_eff = 0.0

        # 2. Cross-modal attention entropy and cosine alignment
        cross_attn = self.compute_cross_modal_attention(v_arr, t_arr)
        if n_v > 1:
            attn_safe = np.maximum(cross_attn, 1e-12)
            row_ents = -np.sum(cross_attn * np.log(attn_safe), axis=1) / math.log(float(n_v))
            cm_entropy = float(np.clip(np.mean(row_ents), 0.0, 1.0))
        else:
            cm_entropy = 0.0

        v_unit = v_arr / np.maximum(np.linalg.norm(v_arr, axis=1, keepdims=True), 1e-12)
        t_unit = t_arr / np.maximum(np.linalg.norm(t_arr, axis=1, keepdims=True), 1e-12)
        cos_matrix = t_unit @ v_unit.T
        max_cos_per_query = np.max(cos_matrix, axis=1)
        cm_alignment = float(np.clip(0.5 * (np.mean(max_cos_per_query) + 1.0), 0.0, 1.0))

        # 3. FastV + PruMerge patch compression
        _, retained_idx, pruned_idx, recon_err = self.prune_and_merge_patches(
            v_arr, t_arr, retention_ratio=retention_ratio
        )
        actual_ret_ratio = float(len(retained_idx) / float(n_v))

        # 4. Linguistic complexity + visual structural complexity
        pooled_text = np.mean(t_arr, axis=0, keepdims=True)
        if text_probe is not None and text_probe.get_params().get("fitted", False):
            # Adapt dimension if text_probe feature_dim differs
            expected_d = int(text_probe.weights.shape[0])
            if pooled_text.shape[1] == expected_d:
                text_prob = float(text_probe.predict_proba(pooled_text)[0])
            else:
                repeats = int(math.ceil(expected_d / float(pooled_text.shape[1])))
                tiled = np.tile(pooled_text, (1, repeats))[:, :expected_d]
                text_prob = float(text_probe.predict_proba(tiled)[0])
        else:
            text_norm = float(np.linalg.norm(pooled_text)) / math.sqrt(float(d_dim))
            text_prob = float(1.0 / (1.0 + math.exp(-(text_norm - 1.0))))

        visual_complexity = float(
            np.clip(0.50 * norm_d_eff + 0.35 * cm_entropy + 0.15 * recon_err, 0.0, 1.0)
        )
        prob_complex = float(
            np.clip(
                self.text_weight * text_prob + self.vision_weight * visual_complexity,
                0.0,
                1.0,
            )
        )

        return MultiModalProbeTelemetry(
            num_vision_patches=n_v,
            num_text_tokens=n_t,
            vision_effective_dim=round(d_eff_v, 6),
            cross_modal_alignment=round(cm_alignment, 6),
            cross_modal_entropy=round(cm_entropy, 6),
            prob_complex=round(prob_complex, 6),
            retained_patch_indices=retained_idx,
            pruned_patch_indices=pruned_idx,
            patch_retention_ratio=round(actual_ret_ratio, 6),
            prumerge_reconstruction_error=recon_err,
        )

    @staticmethod
    def synthesize_vision_patches_from_text_activation(
        activation: np.ndarray,
        num_patches: int = 16,
        image_complexity_hint: float = 0.5,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decomposes a 1D residual vector into deterministic (vision_patches, text_tokens)."""
        if num_patches < 2:
            raise ValueError(f"num_patches must be >= 2, got {num_patches}.")
        h_vec = np.asarray(activation, dtype=np.float64).ravel()
        d_sub = max(8, int(h_vec.size // 4))
        base_text = (
            h_vec[: d_sub * 4].reshape(4, d_sub) if h_vec.size >= d_sub * 4 else h_vec[None, :]
        )
        feat_dim = int(base_text.shape[1])

        rng = np.random.default_rng(seed)
        hint = float(np.clip(image_complexity_hint, 0.0, 1.0))
        anchor = np.mean(base_text, axis=0)
        low_rank_basis = rng.normal(0.0, 0.3, size=(2, feat_dim))
        high_rank_noise = rng.normal(0.0, 1.0, size=(num_patches, feat_dim))
        weights = rng.uniform(0.1, 1.0, size=(num_patches, 2))

        # Blend low-rank anchor structure and high-rank visual dispersion according to hint
        patches = (1.0 - hint) * (
            anchor[None, :] + weights @ low_rank_basis
        ) + hint * high_rank_noise
        return patches.astype(np.float64), base_text.astype(np.float64)


__all__ = [
    "MultiModalActivationProbe",
    "MultiModalProbeTelemetry",
]
