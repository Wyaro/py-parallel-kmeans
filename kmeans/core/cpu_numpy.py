# core/cpu_numpy.py
from __future__ import annotations

import numpy as np

from .base import KMeansBase


class KMeansCPUNumpy(KMeansBase):
    """Простая однопоточная реализация KMeans на NumPy (baseline)."""

    def assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        # (N, K, D) → (N, K)
        diff = X[:, None, :] - centroids[None, :, :]
        distances = np.sum(diff * diff, axis=2)
        return np.argmin(distances, axis=1)

    def update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        D = X.shape[1]
        centroids = np.zeros((self.K, D), dtype=np.float64)

        for k in range(self.K):
            points = X[labels == k]
            if len(points) > 0:
                centroids[k] = points.mean(axis=0)

        return centroids
