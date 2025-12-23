"""
CUDA/CuPy реализации KMeans для бенчмаркинга.

Варианты:
- KMeansGPUCuPy: базовая версия с явным вычислением diff (просто и наглядно).
- KMeansGPUCuPyBincount: версия с редукцией через bincount, без материализации diff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

try:  # CuPy опционален: можем работать без GPU
    import cupy as cp

    _GPU_OK = True
except Exception:  # noqa: BLE001
    cp = None  # type: ignore
    _GPU_OK = False

import numpy as np

from kmeans.core.base import KMeansBase
from kmeans.metrics.timers import Timer


def gpu_available() -> bool:
    """Проверка доступности CuPy/CUDA."""
    return _GPU_OK


class DistanceKernel(Protocol):
    """Протокол ядра вычисления расстояний (для читаемости)."""

    def __call__(self, X: "cp.ndarray", centroids: "cp.ndarray") -> "cp.ndarray":
        ...


@dataclass(frozen=True)
class _GPUArrays:
    """Контейнер для часто используемых GPU-массивов."""

    X: "cp.ndarray"
    centroids: "cp.ndarray"


class _KMeansGPUBase(KMeansBase):
    """
    Общий каркас для GPU-вариантов.

    - Один перенос данных на GPU в fit.
    - Хранит копии результатов на CPU (centroids_cpu/labels_cpu) для последующего анализа.
    """

    def __init__(
        self, n_clusters: int, n_iters: int = 100, tol: float = 1e-6, logger: Any | None = None
    ):
        if not _GPU_OK:
            raise RuntimeError("CuPy/CUDA недоступен, GPU KMeans выключен")
        super().__init__(n_clusters=n_clusters, n_iters=n_iters, tol=tol, logger=logger)
        self._gpu_data: _GPUArrays | None = None

    def _to_gpu(self, X: np.ndarray) -> "cp.ndarray":
        return cp.asarray(X, dtype=cp.float64, order="C")

    def _to_host(self, arr: "cp.ndarray") -> np.ndarray:
        return cp.asnumpy(arr)

    def fit(self, X: np.ndarray, initial_centroids: np.ndarray) -> None:
        # Один раз переносим на GPU: это основная плата за H2D.
        X_gpu = self._to_gpu(X)
        init_gpu = self._to_gpu(initial_centroids)
        self._gpu_data = _GPUArrays(X=X_gpu, centroids=init_gpu)

        # Переопределяем fit для GPU массивов с правильной проверкой сходимости
        self.centroids = init_gpu.copy()

        # сбрасываем накопленные тайминги для нового запуска
        self.t_assign_total = 0.0
        self.t_update_total = 0.0
        self.t_iter_total = 0.0
        self.n_iters_actual = 0

        for i in range(self.n_iters):
            # Сохраняем старые центроиды для проверки сходимости
            old_centroids = self.centroids.copy()

            with Timer() as t_assign:
                self.labels = self.assign_clusters(X_gpu, self.centroids)
            with Timer() as t_update:
                new_centroids = self.update_centroids(X_gpu, self.labels)

            t_assign_elapsed = t_assign.elapsed
            t_update_elapsed = t_update.elapsed
            t_iter_elapsed = t_assign_elapsed + t_update_elapsed

            self.t_assign_total += t_assign_elapsed
            self.t_update_total += t_update_elapsed
            self.t_iter_total += t_iter_elapsed
            self.n_iters_actual = i + 1

            # Проверка сходимости для GPU массивов (используем CuPy функции)
            max_change = float(cp.max(cp.abs(new_centroids - old_centroids)))
            converged = max_change < self.tol

            if self.logger and (i == 0 or (i + 1) % 10 == 0 or converged):
                status = " (converged)" if converged else ""
                self.logger.info(
                    f"  Iteration {i + 1}/{self.n_iters}{status} "
                    f"(T_assign={t_assign_elapsed:.6f}s, "
                    f"T_update={t_update_elapsed:.6f}s, "
                    f"max_change={max_change:.2e})"
                )

            # Обновляем центроиды
            self.centroids = new_centroids

            # Ранний выход при сходимости
            if converged:
                if self.logger:
                    self.logger.info(
                        f"  Convergence reached after {i + 1} iterations "
                        f"(max_change={max_change:.2e} < tol={self.tol:.2e})"
                    )
                break

        # Сохраняем копии на CPU для потенциальной пост-обработки/отладки.
        self.centroids_cpu = self._to_host(self.centroids)
        self.labels_cpu = self._to_host(self.labels)

    # --- Вспомогательное: обработка пустых кластеров оставляем прежней логики ---
    def _merge_centroids(self, new_centroids: "cp.ndarray", non_empty: "cp.ndarray") -> "cp.ndarray":
        """
        Возвращает центроиды, в которых пустые кластеры сохраняют прежние координаты.
        """
        if self.centroids is None:
            return new_centroids
        return cp.where(non_empty[:, None], new_centroids, self.centroids)


class KMeansGPUCuPy(_KMeansGPUBase):
    """
    Базовая реализация через CuPy.

    Прямое броадкаст-вычисление diff: просто и хорошо читается, но требует памяти O(N*K*D).
    Подходит для умеренных N/K/D и для чистого сравнения с CPU baseline.
    """

    def assign_clusters(self, X: "cp.ndarray", centroids: "cp.ndarray") -> "cp.ndarray":
        diff = X[:, None, :] - centroids[None, :, :]  # (N, K, D)
        distances = cp.einsum("nkd,nkd->nk", diff, diff, optimize=True)
        return cp.argmin(distances, axis=1).astype(cp.int32, copy=False)

    def update_centroids(self, X: "cp.ndarray", labels: "cp.ndarray") -> "cp.ndarray":
        K = self.K
        D = X.shape[1]
        sums = cp.zeros((K, D), dtype=cp.float64)
        # CuPy scatter_add (add.at) не поддерживает int64, используем int32.
        counts = cp.zeros(K, dtype=cp.int32)

        # add.at — компактная и наглядная редукция
        cp.add.at(sums, labels, X)
        cp.add.at(counts, labels, 1)

        new_centroids = cp.zeros_like(sums)
        non_empty = counts > 0
        new_centroids[non_empty] = sums[non_empty] / counts[non_empty, None]

        return self._merge_centroids(new_centroids, non_empty)


class KMeansGPUCuPyBincount(_KMeansGPUBase):
    """
    Версия с редукцией через bincount.

    Отличия:
    - assign: матричная формула без материализации diff (меньше памяти).
    - update: редукция сумм по признакам через bincount.
    """

    def assign_clusters(self, X: "cp.ndarray", centroids: "cp.ndarray") -> "cp.ndarray":
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c
        x_sq = cp.sum(X * X, axis=1, keepdims=True)  # (N, 1)
        c_sq = cp.sum(centroids * centroids, axis=1)  # (K,)
        cross = X @ centroids.T  # (N, K)
        distances = x_sq + c_sq[None, :] - 2.0 * cross
        return cp.argmin(distances, axis=1).astype(cp.int32, copy=False)

    def update_centroids(self, X: "cp.ndarray", labels: "cp.ndarray") -> "cp.ndarray":
        K = self.K
        D = X.shape[1]

        counts = cp.bincount(labels, minlength=K).astype(cp.int64, copy=False)
        sums = cp.empty((K, D), dtype=cp.float64)

        # Редукция по каждому признаку отдельно — без scatter-add.
        for d in range(D):
            sums[:, d] = cp.bincount(labels, weights=X[:, d], minlength=K)

        new_centroids = cp.zeros_like(sums)
        non_empty = counts > 0
        new_centroids[non_empty] = sums[non_empty] / counts[non_empty, None]

        return self._merge_centroids(new_centroids, non_empty)