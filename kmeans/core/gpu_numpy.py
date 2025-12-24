"""
CUDA/CuPy реализации KMeans для бенчмаркинга.

Варианты (от медленного к быстрому):
- KMeansGPUCuPyV1: базовая версия с явным вычислением diff (просто и наглядно).
- KMeansGPUCuPyV2: оптимизированная версия с матричной формулой без diff.
- KMeansGPUCuPyV3: быстрая версия с float32 и оптимизациями.
- KMeansGPUCuPyV4: самая быстрая версия с raw CUDA kernels.
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


class KMeansGPUCuPyV1(_KMeansGPUBase):
    """
    Базовая реализация через CuPy (V1 - самая медленная).

    Использует оптимизированную формулу без материализации diff для экономии памяти.
    Подходит для всех размеров датасетов, включая большие N.
    """

    def assign_clusters(self, X: "cp.ndarray", centroids: "cp.ndarray") -> "cp.ndarray":
        # Оптимизированная формула без материализации diff: O(N*K) памяти вместо O(N*K*D)
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
        # CuPy scatter_add (add.at) не поддерживает int64, используем int32.
        counts = cp.zeros(K, dtype=cp.int32)

        # add.at — компактная и наглядная редукция
        cp.add.at(sums, labels, X)
        cp.add.at(counts, labels, 1)

        new_centroids = cp.zeros_like(sums)
        non_empty = counts > 0
        new_centroids[non_empty] = sums[non_empty] / counts[non_empty, None]

        return self._merge_centroids(new_centroids, non_empty)


class KMeansGPUCuPyV2(_KMeansGPUBase):
    """
    Оптимизированная версия быстрее V1 (V2).

    Отличия от V1:
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


class KMeansGPUCuPyV3(_KMeansGPUBase):
    """
    Быстрая GPU-версия быстрее V2 (V3).

    Отличия от V2:
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


class KMeansGPUCuPyV4(_KMeansGPUBase):
    """
    V4 (improved): реально гибридная версия.

    - assign:
        * D <= ASSIGN_RAW_MAX_D: raw-kernel (быстро на очень малых D)
        * иначе: GEMM-формула (как V3), т.к. cuBLAS почти всегда выигрывает на D>=32
      Плюс: x_sq кэшируется (X неизменен в fit).

    - update:
        * small-case (K <= 32, D <= 32, K*D <= 1024): двухпроходная редукция (без глобальных атомиков)
        * иначе: fallback на V3 update (bincount + add.at)

    Важно: никаких synchronize() внутри алгоритма.
    """

    # --- Пороги (эмпирические; вы можете подогнать под свою GPU) ---
    ASSIGN_RAW_MAX_D = 16
    UPDATE_MAX_K = 32
    UPDATE_MAX_D = 32
    UPDATE_MAX_KD = 1024

    _ASSIGN_KERNEL_SMALLD = r"""
    extern "C" __global__
    void assign_smallD(const float* __restrict__ X,
                       const float* __restrict__ C,
                       const float* __restrict__ x_sq,
                       const float* __restrict__ c_sq,
                       int* __restrict__ labels,
                       const int N, const int D, const int K) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= N) return;

        const float* x = X + (size_t)i * (size_t)D;
        float x_norm = x_sq[i];

        float best = 1e30f;
        int best_k = 0;

        // D <= 16 ожидается по диспатчу
        for (int k = 0; k < K; ++k) {
            const float* c = C + (size_t)k * (size_t)D;
            float dot = 0.0f;

            #pragma unroll
            for (int d = 0; d < 16; ++d) {
                if (d < D) dot += x[d] * c[d];
            }

            float dist = x_norm + c_sq[k] - 2.0f * dot;
            if (dist < best) { best = dist; best_k = k; }
        }

        labels[i] = best_k;
    }
    """

    # Kernel 1: частичные суммы/счётчики по блокам без глобальных атомиков
    # partial_sums: [num_blocks, K, D], partial_counts: [num_blocks, K]
    _UPDATE_PARTIAL_KERNEL = r"""
    extern "C" __global__
    void update_partial(const float* __restrict__ X,
                        const int* __restrict__ labels,
                        float* __restrict__ partial_sums,
                        int* __restrict__ partial_counts,
                        const int N, const int D, const int K) {
        int tid = threadIdx.x;
        int bid = blockIdx.x;
        int idx = bid * blockDim.x + tid;

        // Плоские индексы в выходных буферах
        // partial_sums[(bid*K + k)*D + d]
        // partial_counts[bid*K + k]

        // Инициализация partial для этого блока: параллельно по tid
        // (K*D + K) элементов
        int total_f = K * D;
        for (int t = tid; t < total_f; t += blockDim.x) {
            partial_sums[(size_t)bid * (size_t)total_f + (size_t)t] = 0.0f;
        }
        for (int t = tid; t < K; t += blockDim.x) {
            partial_counts[bid * K + t] = 0;
        }
        __syncthreads();

        if (idx >= N) return;

        int k = labels[idx];
        if (k < 0 || k >= K) return;

        // Для small-case разрешаем атомики только в пределах блока (в shared было бы лучше,
        // но здесь мы пишем в global partial буфер, который уникален для блока -> contention ограничен
        // потоками одного блока).
        atomicAdd(&partial_counts[bid * K + k], 1);

        const float* x = X + (size_t)idx * (size_t)D;
        float* ps = partial_sums + ((size_t)bid * (size_t)K + (size_t)k) * (size_t)D;

        for (int d = 0; d < D; ++d) {
            atomicAdd(&ps[d], x[d]);
        }
    }
    """

    # Kernel 2: редукция по блокам -> итоговые sums/counts (без атомиков: один поток на (k,d))
    _UPDATE_REDUCE_KERNEL = r"""
    extern "C" __global__
    void reduce_partials(const float* __restrict__ partial_sums,
                         const int* __restrict__ partial_counts,
                         float* __restrict__ sums,
                         int* __restrict__ counts,
                         const int num_blocks,
                         const int D, const int K) {
        int k = blockIdx.x;      // 0..K-1
        int d = threadIdx.x;     // 0..D-1  (D <= 32 ожидается)

        if (k >= K || d >= D) return;

        float acc = 0.0f;
        int cacc = 0;

        // Суммируем по блокам
        for (int b = 0; b < num_blocks; ++b) {
            acc += partial_sums[((size_t)b * (size_t)K + (size_t)k) * (size_t)D + (size_t)d];
        }

        // counts считаем только одним d==0 потоком
        if (d == 0) {
            for (int b = 0; b < num_blocks; ++b) {
                cacc += partial_counts[b * K + k];
            }
            counts[k] = cacc;
        }

        sums[(size_t)k * (size_t)D + (size_t)d] = acc;
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

        self._assign_smallD = cp.RawKernel(self._ASSIGN_KERNEL_SMALLD, "assign_smallD")
        self._update_partial = cp.RawKernel(self._UPDATE_PARTIAL_KERNEL, "update_partial")
        self._reduce_partials = cp.RawKernel(self._UPDATE_REDUCE_KERNEL, "reduce_partials")

        # cache (valid within one fit run)
        self._x_sq_cache: "cp.ndarray | None" = None

    def _to_gpu(self, X: np.ndarray) -> "cp.ndarray":
        return cp.asarray(X, dtype=self._dtype, order="C")

    def fit(self, X: np.ndarray, initial_centroids: np.ndarray) -> None:
        # сбрасываем cache на новый запуск
        self._x_sq_cache = None
        super().fit(X, initial_centroids)

    def _ensure_x_sq(self, X_f32: "cp.ndarray") -> "cp.ndarray":
        if self._x_sq_cache is None or self._x_sq_cache.shape[0] != X_f32.shape[0]:
            self._x_sq_cache = cp.sum(X_f32 * X_f32, axis=1)  # (N,)
        return self._x_sq_cache

    def assign_clusters(self, X: "cp.ndarray", centroids: "cp.ndarray") -> "cp.ndarray":
        N, D = X.shape
        K = self.K

        X_f32 = X.astype(cp.float32, copy=False)
        C_f32 = centroids.astype(cp.float32, copy=False)

        x_sq = self._ensure_x_sq(X_f32)
        c_sq = cp.sum(C_f32 * C_f32, axis=1)  # (K,)

        # Диспатч: raw только для малых D, иначе GEMM (V3 формула)
        if D <= self.ASSIGN_RAW_MAX_D:
            labels = cp.empty(N, dtype=cp.int32)
            threads = 256
            blocks = (N + threads - 1) // threads
            self._assign_smallD(
                (blocks,),
                (threads,),
                (X_f32, C_f32, x_sq, c_sq, labels, N, D, K),
            )
            return labels

        # GEMM-путь (на D>=32 обычно быстрее raw)
        # ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x·c
        cross = X_f32 @ C_f32.T
        distances = x_sq[:, None] + c_sq[None, :] - 2.0 * cross
        return cp.argmin(distances, axis=1).astype(cp.int32, copy=False)

    def update_centroids(self, X: "cp.ndarray", labels: "cp.ndarray") -> "cp.ndarray":
        N, D = X.shape
        K = self.K
        X_f32 = X.astype(cp.float32, copy=False)
        labels_i32 = labels.astype(cp.int32, copy=False)

        # Диспатч: двухпроходная редукция только для small-case
        use_two_pass = (K <= self.UPDATE_MAX_K) and (D <= self.UPDATE_MAX_D) and (K * D <= self.UPDATE_MAX_KD)

        if not use_two_pass:
            # Fallback на стабильный V3 update
            counts = cp.bincount(labels_i32, minlength=K).astype(cp.int32, copy=False)
            sums = cp.zeros((K, D), dtype=cp.float32)
            cp.add.at(sums, labels_i32, X_f32)
        else:
            threads = 256
            blocks = (N + threads - 1) // threads
            num_blocks = int(blocks)

            # partial buffers
            partial_sums = cp.empty((num_blocks, K, D), dtype=cp.float32)
            partial_counts = cp.empty((num_blocks, K), dtype=cp.int32)

            self._update_partial(
                (num_blocks,),
                (threads,),
                (X_f32, labels_i32, partial_sums, partial_counts, N, D, K),
            )

            sums = cp.empty((K, D), dtype=cp.float32)
            counts = cp.empty((K,), dtype=cp.int32)

            # grid: K блоков; threads: D (<=32 по диспатчу)
            self._reduce_partials(
                (K,),
                (max(32, D),),  # безопасно: лишние потоки сами выйдут
                (partial_sums, partial_counts, sums, counts, num_blocks, D, K),
            )

        new_centroids = cp.zeros_like(sums)
        non_empty = counts > 0
        new_centroids[non_empty] = sums[non_empty] / counts[non_empty, None]
        return self._merge_centroids(new_centroids, non_empty)