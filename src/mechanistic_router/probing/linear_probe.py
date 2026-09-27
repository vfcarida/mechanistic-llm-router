"""Linear and Logistic Activation Probes for residual stream features."""

from __future__ import annotations

from pathlib import Path
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
        self._weights: np.ndarray | None = None
        self._classes: np.ndarray | None = None

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

    @property
    def weights(self) -> np.ndarray:
        """Unnormalized weight vector of the decision boundary (shape: (D,))."""
        if self._weights is None:
            raise ValueError("Probe has not been fitted yet.")
        return self._weights

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
        self._classes = np.array(clf.classes_)

        # Extract normal vector to decision boundary
        w = clf.coef_[0]
        norm = float(np.linalg.norm(w))
        if norm > 0:
            self._direction_vector = (w / norm).astype(np.float32)
        else:
            self._direction_vector = np.zeros_like(w, dtype=np.float32)
        self._norm = norm
        self._intercept = float(clf.intercept_[0])
        self._weights = w.astype(np.float32)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability of class 1 (needs strong model) for each sample (shape: (N,))."""
        if self._weights is not None:
            margin = self.decision_function(X)
            clipped_margin = np.clip(margin, -50.0, 50.0)
            return np.asarray(1.0 / (1.0 + np.exp(-clipped_margin)), dtype=np.float32)
        if self.classifier is None:
            raise ValueError("Probe has not been fitted yet.")
        probs = np.asarray(self.classifier.predict_proba(X))
        return np.asarray(probs[:, 1], dtype=np.float32)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Returns binary predictions using custom decision threshold."""
        probs = self.predict_proba(X)
        return np.asarray(probs >= threshold, dtype=int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Returns raw margin (X @ w + b)."""
        if self._weights is not None:
            X_arr = np.asarray(X, dtype=np.float32)
            margin = np.dot(X_arr, self._weights) + self._intercept
            return np.atleast_1d(np.asarray(margin, dtype=np.float32))
        if self.classifier is None:
            raise ValueError("Probe has not been fitted yet.")
        margin = np.asarray(self.classifier.decision_function(X))
        return np.atleast_1d(np.asarray(margin, dtype=np.float32))

    def save(self, filepath: str | Path) -> None:
        """Saves probe weights, hyperparameters, and decision hyperplane parameters to disk.

        Uses compressed NumPy format (.npz) to avoid arbitrary code execution
        risks associated with standard pickle serialization.
        """
        if self._weights is None or self._direction_vector is None:
            raise ValueError("Cannot save an unfitted probe.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        classes = self._classes if self._classes is not None else np.array([0, 1], dtype=int)

        np.savez_compressed(
            str(path),
            weights=self._weights,
            direction_vector=self._direction_vector,
            norm=np.array([self._norm], dtype=np.float32),
            intercept=np.array([self._intercept], dtype=np.float32),
            hyperparams=np.array(
                [self.C, float(self.random_state), float(self.max_iter)],
                dtype=np.float64,
            ),
            classes=classes,
        )

    @classmethod
    def load(cls, filepath: str | Path) -> LinearActivationProbe:
        """Loads a fitted probe from a compressed NumPy archive (.npz).

        Does not execute arbitrary code (unlike pickle), ensuring safe loading in production.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Probe weights archive not found at: {path}")

        with np.load(str(path)) as data:
            weights = data["weights"]
            direction_vector = data["direction_vector"]
            norm = float(data["norm"][0])
            intercept = float(data["intercept"][0])
            hyperparams = data["hyperparams"]
            classes = data["classes"] if "classes" in data else np.array([0, 1])
            c_val = float(hyperparams[0])
            random_state = int(hyperparams[1])
            max_iter = int(hyperparams[2])

        probe = cls(C=c_val, random_state=random_state, max_iter=max_iter)
        probe._weights = weights
        probe._direction_vector = direction_vector
        probe._norm = norm
        probe._intercept = intercept
        probe._classes = classes

        try:
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(
                C=probe.C, random_state=probe.random_state, max_iter=probe.max_iter
            )
            clf.coef_ = weights.reshape(1, -1)
            clf.intercept_ = np.array([intercept])
            clf.classes_ = classes
            probe.classifier = clf
        except ImportError:
            probe.classifier = None

        return probe

    def get_params(self) -> dict[str, Any]:
        """Returns probe parameter configuration."""
        return {
            "C": self.C,
            "random_state": self.random_state,
            "max_iter": self.max_iter,
            "norm": self._norm,
            "intercept": self._intercept,
            "fitted": self._weights is not None,
        }


__all__ = ["LinearActivationProbe"]
