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
        self.t_h2d: float = 0.0
        self.t_d2h: float = 0.0

    def _to_gpu(self, X: np.ndarray) -> "cp.ndarray":
        return cp.asarray(X, dtype=cp.float64, order="C")

    def _to_host(self, arr: "cp.ndarray") -> np.ndarray:
        return cp.asnumpy(arr)

    def fit(self, X: np.ndarray, initial_centroids: np.ndarray) -> None:
        # Один раз переносим на GPU: фиксируем время передачи.
        with Timer() as t_h2d:
            X_gpu = self._to_gpu(X)
            init_gpu = self._to_gpu(initial_centroids)
        self.t_h2d = float(t_h2d.elapsed)
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
        with Timer() as t_d2h:
            self.centroids_cpu = self._to_host(self.centroids)
            self.labels_cpu = self._to_host(self.labels)
        self.t_d2h = float(t_d2h.elapsed)

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
    Оптимизированная версия быстрее базовой.

    Отличия от базовой версии:
    - assign: матричная формула ||x||² + ||c||² - 2x·c без материализации diff (меньше памяти O(N*K) вместо O(N*K*D)).
    - update: оптимизированная редукция через scatter_add (cp.add.at).
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
        sums = cp.zeros((K, D), dtype=cp.float64)
        counts = cp.zeros(K, dtype=cp.int32)

        # Оптимизированная редукция через scatter_add (быстрее цикла по D)
        # Это делает bincount быстрее базовой версии за счет оптимизированного assign
        cp.add.at(sums, labels, X)
        cp.add.at(counts, labels, 1)

        new_centroids = cp.zeros_like(sums)
        non_empty = counts > 0
        new_centroids[non_empty] = sums[non_empty] / counts[non_empty, None]

        return self._merge_centroids(new_centroids, non_empty)


class KMeansGPUCuPyFast(_KMeansGPUBase):
    """
    Быстрая GPU-версия быстрее bincount.

    Отличия от bincount:
    - assign: та же оптимизированная формула без diff.
    - update: оптимизированная редукция через scatter_add.
    - float32 по умолчанию (меньше трафик и память, быстрее вычисления).
    """

    def __init__(
        self,
        n_clusters: int,
        n_iters: int = 100,
        tol: float = 1e-6,
        use_float32: bool = True,
        logger: Any | None = None,
    ) -> None:
        super().__init__(n_clusters=n_clusters, n_iters=n_iters, tol=tol, logger=logger)
        self._dtype = cp.float32 if use_float32 else cp.float64

    def _to_gpu(self, X: np.ndarray) -> "cp.ndarray":
        return cp.asarray(X, dtype=self._dtype, order="C")

    def assign_clusters(self, X: "cp.ndarray", centroids: "cp.ndarray") -> "cp.ndarray":
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c
        x_sq = cp.sum(X * X, axis=1, keepdims=True)            # (N, 1)
        c_sq = cp.sum(centroids * centroids, axis=1)           # (K,)
        cross = X @ centroids.T                                # (N, K)
        distances = x_sq + c_sq[None, :] - 2.0 * cross
        return cp.argmin(distances, axis=1).astype(cp.int32, copy=False)

    def update_centroids(self, X: "cp.ndarray", labels: "cp.ndarray") -> "cp.ndarray":
        K = self.K
        D = X.shape[1]

        counts = cp.bincount(labels, minlength=K).astype(cp.int32, copy=False)
        sums = cp.zeros((K, D), dtype=cp.float64 if self._dtype == cp.float64 else cp.float32)

        # Оптимизированная редукция через scatter_add вместо цикла по D
        # Это быстрее для больших D, так как один вызов вместо D вызовов
        cp.add.at(sums, labels, X)

        new_centroids = cp.zeros_like(sums)
        non_empty = counts > 0
        new_centroids[non_empty] = sums[non_empty] / counts[non_empty, None]

        return self._merge_centroids(new_centroids, non_empty)


class KMeansGPUCuPyRaw(_KMeansGPUBase):
    """
    Самая быстрая raw-kernel версия.

    Отличия от fast:
    - assign: raw CUDA kernel с ручной оптимизацией (без materialize diff/GEMM).
    - update: raw CUDA kernel с атомарными операциями для максимальной производительности.
    - float32 по умолчанию для меньшего трафика и памяти.
    Подходит при умеренных K (например, K<=64) и средних D; не требует больших временных буферов.
    """

    _ASSIGN_KERNEL = r"""
    extern "C" __global__
    void assign_kernel(const float* __restrict__ X,
                       const float* __restrict__ C,
                       int* __restrict__ labels,
                       const int N, const int D, const int K) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= N) return;
        const float* x = X + i * D;
        float best = 1e30f;
        int best_k = 0;
        for (int k = 0; k < K; ++k) {
            const float* c = C + k * D;
            float dist = 0.0f;
            #pragma unroll 4
            for (int d = 0; d < D; ++d) {
                float diff = x[d] - c[d];
                dist += diff * diff;
            }
            if (dist < best) { best = dist; best_k = k; }
        }
        labels[i] = best_k;
    }
    """
    
    _UPDATE_KERNEL = r"""
    extern "C" __global__
    void update_kernel(const float* __restrict__ X,
                       const int* __restrict__ labels,
                       float* __restrict__ sums,
                       int* __restrict__ counts,
                       const int N, const int D, const int K) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= N) return;
        
        int label = labels[idx];
        if (label < 0 || label >= K) return;
        
        // Атомарное добавление к счетчикам (int)
        atomicAdd(&counts[label], 1);
        
        // Атомарное добавление к суммам (float)
        // atomicAdd для float доступен на compute capability 2.0+
        for (int d = 0; d < D; ++d) {
            atomicAdd(&sums[label * D + d], X[idx * D + d]);
        }
    }
    """

    def __init__(
        self,
        n_clusters: int,
        n_iters: int = 100,
        tol: float = 1e-6,
        use_float32: bool = True,
        logger: Any | None = None,
    ) -> None:
        super().__init__(n_clusters=n_clusters, n_iters=n_iters, tol=tol, logger=logger)
        self._dtype = cp.float32 if use_float32 else cp.float64
        # компилируем кернелы один раз
        self._assign_kernel = cp.RawKernel(self._ASSIGN_KERNEL, "assign_kernel")
        self._update_kernel = cp.RawKernel(self._UPDATE_KERNEL, "update_kernel")

    def _to_gpu(self, X: np.ndarray) -> "cp.ndarray":
        return cp.asarray(X, dtype=self._dtype, order="C")

    def assign_clusters(self, X: "cp.ndarray", centroids: "cp.ndarray") -> "cp.ndarray":
        N, D = X.shape
        K = self.K
        labels = cp.empty(N, dtype=cp.int32)

        # запускаем один кернел для всего набора
        threads = 256
        blocks = (N + threads - 1) // threads
        self._assign_kernel(
            (blocks,),
            (threads,),
            (X.astype(cp.float32, copy=False),  # гарантируем float32 для ядра
             centroids.astype(cp.float32, copy=False),
             labels,
             N, D, K),
        )
        return labels

    def update_centroids(self, X: "cp.ndarray", labels: "cp.ndarray") -> "cp.ndarray":
        N, D = X.shape
        K = self.K
        
        # Используем raw CUDA kernel для максимальной производительности
        # Raw kernel использует float32 (hardcoded), поэтому приводим к float32
        X_f32 = X.astype(cp.float32, copy=False)
        sums = cp.zeros((K, D), dtype=cp.float32)
        counts = cp.zeros(K, dtype=cp.int32)
        
        # Запускаем raw kernel для редукции
        threads = 256
        blocks = (N + threads - 1) // threads
        self._update_kernel(
            (blocks,),
            (threads,),
            (X_f32,
             labels.astype(cp.int32, copy=False),
             sums,
             counts,
             N, D, K),
        )
        
        # Синхронизируем перед вычислением центроидов
        # Ждем завершения всех операций на GPU
        # В CuPy операции обычно синхронные, но для raw kernel нужна явная синхронизация
        # Используем getDevice() для получения текущего устройства и синхронизации
        device = cp.cuda.Device()
        device.synchronize()
        
        new_centroids = cp.zeros_like(sums)
        non_empty = counts > 0
        new_centroids[non_empty] = sums[non_empty] / counts[non_empty, None]

        return self._merge_centroids(new_centroids, non_empty)