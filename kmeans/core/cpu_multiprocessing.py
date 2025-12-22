from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool, RawArray, cpu_count
from typing import Optional, Tuple, List

import numpy as np

from kmeans.core.base import KMeansBase


@dataclass(frozen=True)
class MultiprocessingConfig:
    """
    Конфигурация многопроцессорной реализации.

    n_processes: желаемое число процессов (ядёр CPU).
    chunk_size: размер батча по объектам; если None — N делится равномерно по процессам.
    """

    n_processes: int = 4
    chunk_size: Optional[int] = None


# --- Глобальное состояние для доступа к разделяемому X в воркерах ---

_SHARED_X_BUF: RawArray | None = None
_SHARED_X_SHAPE: Tuple[int, int] | None = None


def _init_shared_X(raw: RawArray, shape: Tuple[int, int]) -> None:
    """Инициализатор пула: настраивает глобальную ссылку на разделяемый X."""
    global _SHARED_X_BUF, _SHARED_X_SHAPE
    _SHARED_X_BUF = raw
    _SHARED_X_SHAPE = shape


def _get_shared_X() -> np.ndarray:
    """Возвращает NumPy-представление разделяемого массива X."""
    assert _SHARED_X_BUF is not None and _SHARED_X_SHAPE is not None
    arr = np.frombuffer(_SHARED_X_BUF, dtype=np.float64)
    return arr.reshape(_SHARED_X_SHAPE)


def _assign_chunk_worker(args: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Рабочая функция для шага назначения (используется в пуле процессов).
    Получает индексы чанка и центроиды, берёт X из разделяемой памяти.
    """
    idx, centroids = args
    X = _get_shared_X()
    X_chunk = X[idx]
    diff = X_chunk[:, None, :] - centroids[None, :, :]
    distances = np.einsum("mkd,mkd->mk", diff, diff, optimize=True)
    return np.argmin(distances, axis=1).astype(np.int32, copy=False)


def _partial_reduce_worker(
    args: Tuple[np.ndarray, np.ndarray, int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Рабочая функция для частичной редукции по кластеру.
    Возвращает (sums[K,D], counts[K]) для своего чанка.
    """
    idx, labels_chunk, K, D = args
    X = _get_shared_X()
    X_chunk = X[idx]

    sums = np.zeros((K, D), dtype=np.float64)
    counts = np.zeros(K, dtype=np.int64)

    for k in range(K):
        mask = labels_chunk == k
        if not np.any(mask):
            continue
        pts = X_chunk[mask]
        sums[k] += pts.sum(axis=0)
        counts[k] += pts.shape[0]

    return sums, counts


class KMeansCPUMultiprocessing(KMeansBase):
    """
    K-Means на CPU с использованием multiprocessing.Pool.

    - assign_clusters: параллелим по чанкам объектов;
    - update_centroids: параллельная частичная редукция, затем слияние.

    Важно: для реальных бенчмарков лучше было бы переиспользовать пул между
    итерациями fit, но для простоты здесь пул создаётся на каждый вызов
    assign/update. Архитектура класса позволяет доработать это позже.
    """

    def __init__(
        self,
        n_clusters: int,
        n_iters: int = 100,
        mp: MultiprocessingConfig = MultiprocessingConfig(),
        logger=None,
    ) -> None:
        super().__init__(n_clusters=n_clusters, n_iters=n_iters, logger=logger)
        self.mp = mp

        # Пул процессов и разбиение на чанки переиспользуются в рамках одного fit,
        # чтобы не платить оверхед создания Pool на каждую итерацию.
        self._pool: Optional[Pool] = None
        self._chunks: Optional[List[np.ndarray]] = None

    # --- Внутренняя инфраструктура пула/чанков ---

    def _ensure_pool_and_chunks(self, X: np.ndarray) -> None:
        """Инициализирует пул процессов, разделяемый X и разбиение индексов, если ещё не сделано."""
        if self._pool is not None and self._chunks is not None:
            return

        n_procs = max(1, min(int(self.mp.n_processes), cpu_count()))
        N = X.shape[0]

        # Разбиение индексов на чанки
        if self.mp.chunk_size is None:
            chunks = np.array_split(np.arange(N), n_procs)
        else:
            cs = int(self.mp.chunk_size)
            if cs <= 0:
                raise ValueError("chunk_size must be positive")
            chunks = [np.arange(i, min(i + cs, N)) for i in range(0, N, cs)]

        # фильтруем пустые чанки заранее
        self._chunks = [idx for idx in chunks if idx.size > 0]

        # создаём разделяемый буфер под X и копируем данные один раз
        X_c = np.ascontiguousarray(X, dtype=np.float64)
        raw = RawArray("d", int(X_c.size))
        shared_view = np.frombuffer(raw, dtype=np.float64).reshape(X_c.shape)
        shared_view[:] = X_c

        # создаём пул, который в каждом процессе инициализирует ссылку на shared X
        self._pool = Pool(
            processes=n_procs,
            initializer=_init_shared_X,
            initargs=(raw, X_c.shape),
        )

    def _close_pool(self) -> None:
        """Закрывает пул после завершения fit."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
        self._pool = None
        self._chunks = None

    # ---------- Assignment (parallel over chunks) ----------

    def assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        self._ensure_pool_and_chunks(X)
        assert self._pool is not None and self._chunks is not None

        labels = np.empty(N, dtype=np.int32)

        # формируем список аргументов: (индексы чанка, centroids)
        args: List[Tuple[np.ndarray, np.ndarray]] = [
            (idx, centroids) for idx in self._chunks
        ]

        results = self._pool.map(_assign_chunk_worker, args)

        for idx, lbl_chunk in zip(self._chunks, results):
            labels[idx] = lbl_chunk

        return labels

    # ---------- Update (parallel reduction) ----------

    def update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        N, D = X.shape
        K = self.K

        self._ensure_pool_and_chunks(X)
        assert self._pool is not None and self._chunks is not None

        args_list: List[Tuple[np.ndarray, np.ndarray, int, int]] = [
            (idx, labels[idx], K, D) for idx in self._chunks
        ]

        partials: List[Tuple[np.ndarray, np.ndarray]] = self._pool.map(
            _partial_reduce_worker, args_list
        )

        sums_total = np.zeros((K, D), dtype=np.float64)
        counts_total = np.zeros(K, dtype=np.int64)

        for sums, counts in partials:
            sums_total += sums
            counts_total += counts

        new_centroids = self.centroids.copy()
        non_empty = counts_total > 0
        new_centroids[non_empty] = sums_total[non_empty] / counts_total[non_empty, None]

        return new_centroids

    # Переопределяем fit, чтобы управлять временем жизни пула

    def fit(self, X: np.ndarray, initial_centroids: np.ndarray) -> None:
        """
        Переопределённый fit:
        - создаёт пул один раз на весь запуск;
        - переиспользует его между итерациями;
        - затем аккуратно закрывает.
        """
        try:
            super().fit(X, initial_centroids)
        finally:
            # независимо от исхода fit корректно закрываем пул
            self._close_pool()



