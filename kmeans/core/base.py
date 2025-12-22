from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from kmeans.metrics.timers import Timer


class KMeansBase(ABC):
    """
    Базовый класс для реализаций KMeans.

    Отвечает за цикл итераций и сбор низкоуровневых таймингов:
    - T_назначения: время шага assign_clusters;
    - T_обновления: время шага update_centroids;
    - T_итерации: сумма двух предыдущих.
    """

    def __init__(self, n_clusters: int, n_iters: int = 100, logger: Any | None = None):
        self.K = n_clusters
        self.n_iters = n_iters
        self.logger = logger

        self.centroids: np.ndarray | None = None
        self.labels: np.ndarray | None = None

        # агрегированные тайминги за один вызов fit(...)
        self.t_assign_total: float = 0.0
        self.t_update_total: float = 0.0
        self.t_iter_total: float = 0.0

    def fit(self, X: np.ndarray, initial_centroids: np.ndarray) -> None:
        """
        Основной цикл KMeans.

        Собирает тайминги по шагам назначения и обновления центроидов.
        """
        self.centroids = initial_centroids.copy()

        # сбрасываем накопленные тайминги для нового запуска
        self.t_assign_total = 0.0
        self.t_update_total = 0.0
        self.t_iter_total = 0.0

        for i in range(self.n_iters):
            with Timer() as t_assign:
                self.labels = self.assign_clusters(X, self.centroids)
            with Timer() as t_update:
                self.centroids = self.update_centroids(X, self.labels)

            t_assign_elapsed = t_assign.elapsed
            t_update_elapsed = t_update.elapsed
            t_iter_elapsed = t_assign_elapsed + t_update_elapsed

            self.t_assign_total += t_assign_elapsed
            self.t_update_total += t_update_elapsed
            self.t_iter_total += t_iter_elapsed

            if self.logger and (i == 0 or (i + 1) % 10 == 0):
                self.logger.info(
                    f"  Iteration {i + 1}/{self.n_iters} "
                    f"(T_assign={t_assign_elapsed:.6f}s, "
                    f"T_update={t_update_elapsed:.6f}s)"
                )

    @abstractmethod
    def assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Шаг назначения точек кластерам."""
        raise NotImplementedError

    @abstractmethod
    def update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Шаг обновления центроидов по присвоенным меткам."""
        raise NotImplementedError
