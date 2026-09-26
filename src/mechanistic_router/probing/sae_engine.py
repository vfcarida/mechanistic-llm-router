from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAEEngine(nn.Module):
    """Sparse Autoencoder (SAE) Engine for Mechanistic Circuit Identification.

    Extracts high-dimensional sparse feature activations from model prefill representations.
    Supports standard ReLU sparsity and TopK activation functions.
    Maps active feature indices to specific cognitive burdens (e.g., polysemantic mathematical
    reasoning circuits vs trivial factual retrieval) to programmatically mutate router decisions.
    """

    def __init__(
        self,
        d_in: int = 128,
        d_sae: int = 512,
        l1_coefficient: float = 0.001,
        top_k: int | None = None,
        circuit_feature_map: dict[str, list[int]] | None = None,
    ):
        """Initializes SAE encoder-decoder architecture parameters.

        Args:
            d_in: Input activation hidden dimension size.
            d_sae: Expanded sparse autoencoder dictionary dimension size.
            l1_coefficient: L1 regularization penalty strength.
            top_k: Optional Top-K sparsity parameter (Gao et al. 2024).
            circuit_feature_map: Optional mapping of circuit types to feature index lists.
        """
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.l1_coefficient = l1_coefficient
        self.top_k = top_k
        self.circuit_feature_map = circuit_feature_map

        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) * 0.02)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in) * 0.02)
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes raw activation tensor into sparse feature activations.

        Supports standard ReLU non-linearity and Top-K activation sparsity.
        """
        hidden = torch.matmul(x - self.b_dec, self.W_enc) + self.b_enc
        acts = torch.relu(hidden)

        if self.top_k is not None and self.top_k < acts.shape[-1]:
            topk_vals, topk_indices = torch.topk(acts, self.top_k, dim=-1)
            zeros = torch.zeros_like(acts)
            acts = zeros.scatter(-1, topk_indices, topk_vals)

        return acts

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Reconstructs original activation tensor from sparse feature vectors."""
        return torch.matmul(feature_acts, self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Executes forward pass through SAE.

        Args:
            x: Input activation tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (reconstructed_x, sparse_feature_activations).
        """
        feature_acts = self.encode(x)
        x_recon = self.decode(feature_acts)
        return x_recon, feature_acts

    def compute_loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Calculates reconstruction loss and L1 sparsity penalty.

        Args:
            x: Target activation tensor to reconstruct.

        Returns:
            Dictionary with 'loss', 'reconstruction_loss', and 'l1_loss'.
        """
        x_recon, feature_acts = self.forward(x)
        recon_loss = F.mse_loss(x_recon, x)
        l1_loss = self.l1_coefficient * feature_acts.abs().sum(dim=-1).mean()
        total_loss = recon_loss + l1_loss
        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "l1_loss": l1_loss,
        }

    def extract_active_circuits(self, x: torch.Tensor, threshold: float = 0.1) -> dict[str, Any]:
        """Extracts active feature indices and classifies cognitive burden type.

        Args:
            x: Input prefill activation tensor [seq_len, d_in] or [batch, seq_len, d_in].
            threshold: Minimum activation threshold for feature firing.

        Returns:
            Dictionary containing active feature indices, sparsity ratio, and
            cognitive classification.
        """
        if x.ndim == 3:
            x = x.mean(dim=1)  # Pool sequence dimension -> [batch, d_in]
        if x.ndim == 1:
            x = x.unsqueeze(0)

        with torch.no_grad():
            f_acts = self.encode(x)
            active_mask = f_acts > threshold
            active_indices = torch.nonzero(active_mask, as_tuple=False)[:, 1].tolist()
            unique_indices = sorted(list(set(active_indices)))

            # If a custom circuit feature map was provided, match against it
            if self.circuit_feature_map:
                circuit_scores: dict[str, int] = {}
                for c_name, indices in self.circuit_feature_map.items():
                    circuit_scores[c_name] = len(set(unique_indices).intersection(indices))
                if circuit_scores:
                    circuit_type = max(circuit_scores, key=lambda k: circuit_scores[k])
                else:
                    circuit_type = "factual_retrieval"
            else:
                # Default heuristic patterns for simulated testing
                has_math_reasoning = any(idx % 5 == 0 for idx in unique_indices)
                has_complex_logic = any(idx % 7 == 0 for idx in unique_indices)

                circuit_type = "factual_retrieval"
                if has_math_reasoning and has_complex_logic:
                    circuit_type = "polysemantic_reasoning"
                elif has_math_reasoning:
                    circuit_type = "mathematical_computation"

        return {
            "active_feature_indices": unique_indices[:15],
            "total_active_features": len(unique_indices),
            "sparsity_ratio": float(len(unique_indices) / self.d_sae),
            "circuit_type": circuit_type,
            "requires_oracle": circuit_type
            in (
                "polysemantic_reasoning",
                "mathematical_computation",
            ),
        }
