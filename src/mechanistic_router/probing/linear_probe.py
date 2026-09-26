"""Linear and Logistic Activation Probes for residual stream features."""

from __future__ import annotations

from typing import Any

import numpy as np


class LinearActivationProbe:
    """Linear/Logistic probe trained on residual stream activation vectors."""

    def __init__(self, C: float = 1.0, random_state: int = 42, max_iter: int = 1000) -> None:
        self.C = C
        self.random_state = random_state
        self.max_iter = max_iter
        self.classifier: Any = None
        self._direction_vector: np.ndarray | None = None
        self._norm: float = 0.0
        self._intercept: float = 0.0

    @property
    def direction_vector(self) -> np.ndarray:
        """Unit-norm directional vector in activation space (shape: (D,))."""
        if self._direction_vector is None:
            raise ValueError("Probe has not been fitted yet.")
        return self._direction_vector

    @property
    def direction_norm(self) -> float:
        """L2 norm of unnormalized weight vector."""
        return self._norm

    @property
    def intercept(self) -> float:
        """Intercept of the fitted hyperplane."""
        return self._intercept

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearActivationProbe:
        """Fits logistic regression on activations X (N, D) and binary labels y (N,)."""
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as err:
            raise ImportError(
                "The 'scikit-learn' package is required to fit LinearActivationProbe. "
                "Install it via: pip install 'mechanistic-router[probe]'"
            ) from err

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ValueError(
                f"Expected at least 2 distinct classes to fit probe, found: {unique_classes}"
            )

        clf = LogisticRegression(
            C=self.C,
            random_state=self.random_state,
            max_iter=self.max_iter,
            solver="lbfgs",
        )
        clf.fit(X, y)
        self.classifier = clf

        # Extract normal vector to decision boundary
        w = clf.coef_[0]
        norm = float(np.linalg.norm(w))
        if norm > 0:
            self._direction_vector = (w / norm).astype(np.float32)
        else:
            self._direction_vector = np.zeros_like(w, dtype=np.float32)
        self._norm = norm
        self._intercept = float(clf.intercept_[0])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability of class 1 (needs strong model) for each sample (shape: (N,))."""
        if self.classifier is None:
            raise ValueError("Probe has not been fitted yet.")
        probs = np.asarray(self.classifier.predict_proba(X))
        return np.asarray(probs[:, 1], dtype=np.float32)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Returns binary predictions using custom decision threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Returns raw margin (X @ w + b)."""
        if self.classifier is None:
            raise ValueError("Probe has not been fitted yet.")
        margin = np.asarray(self.classifier.decision_function(X))
        return np.asarray(margin, dtype=np.float32)

    def get_params(self) -> dict[str, Any]:
        """Returns probe parameter configuration."""
        return {
            "C": self.C,
            "random_state": self.random_state,
            "max_iter": self.max_iter,
            "norm": self._norm,
            "intercept": self._intercept,
        }


__all__ = ["LinearActivationProbe"]
