"""
Тесты согласованности между различными реализациями K-means.

Критически важно: все реализации должны давать одинаковые результаты
на одинаковых входных данных.
"""

import numpy as np
import pytest
from kmeans.core.cpu_numpy import KMeansCPUNumpy
from kmeans.core.cpu_multiprocessing import (
    KMeansCPUMultiprocessing,
    MultiprocessingConfig,
)
from kmeans.core.gpu_numpy import (
    gpu_available,
    KMeansGPUCuPyV1,
    KMeansGPUCuPyV2,
    # Алиасы для обратной совместимости
    KMeansGPUCuPy,
    KMeansGPUCuPyBincount,
)


class TestImplementationConsistency:
    """Тесты согласованности между реализациями."""

    def test_cpu_numpy_vs_multiprocessing(self, small_dataset):
        """CPU NumPy и Multiprocessing должны давать одинаковые результаты."""
        X, initial_centroids = small_dataset

        model_numpy = KMeansCPUNumpy(n_clusters=2, n_iters=20)
        model_numpy.fit(X, initial_centroids)

        model_mp = KMeansCPUMultiprocessing(
            n_clusters=2,
            n_iters=20,
            mp=MultiprocessingConfig(n_processes=2),
        )
        model_mp.fit(X, initial_centroids)

        # Центроиды должны быть близки (порядок может отличаться)
        # Сортируем по первой координате для сравнения
        centroids_numpy_sorted = model_numpy.centroids[
            np.argsort(model_numpy.centroids[:, 0])
        ]
        centroids_mp_sorted = model_mp.centroids[np.argsort(model_mp.centroids[:, 0])]

        np.testing.assert_allclose(
            centroids_numpy_sorted,
            centroids_mp_sorted,
            rtol=1e-5,
            atol=1e-5,
            err_msg="CPU NumPy и Multiprocessing дают разные центроиды",
        )

    @pytest.mark.skipif(not gpu_available(), reason="GPU недоступен")
    def test_cpu_vs_gpu_cupy(self, small_dataset):
        """CPU и GPU реализации должны давать одинаковые результаты."""
        X, initial_centroids = small_dataset

        model_cpu = KMeansCPUNumpy(n_clusters=2, n_iters=20)
        model_cpu.fit(X, initial_centroids)

        model_gpu = KMeansGPUCuPyV1(n_clusters=2, n_iters=20)
        model_gpu.fit(X, initial_centroids)

        # Сравниваем центроиды на CPU
        centroids_cpu_sorted = model_cpu.centroids[
            np.argsort(model_cpu.centroids[:, 0])
        ]
        centroids_gpu_sorted = model_gpu.centroids_cpu[
            np.argsort(model_gpu.centroids_cpu[:, 0])
        ]

        np.testing.assert_allclose(
            centroids_cpu_sorted,
            centroids_gpu_sorted,
            rtol=1e-5,
            atol=1e-5,
            err_msg="CPU и GPU CuPy дают разные центроиды",
        )

    @pytest.mark.skipif(not gpu_available(), reason="GPU недоступен")
    def test_gpu_cupy_vs_bincount(self, small_dataset):
        """Две GPU реализации должны давать одинаковые результаты."""
        X, initial_centroids = small_dataset

        model_cupy = KMeansGPUCuPyV1(n_clusters=2, n_iters=20)
        model_cupy.fit(X, initial_centroids)

        model_bincount = KMeansGPUCuPyV2(n_clusters=2, n_iters=20)
        model_bincount.fit(X, initial_centroids)

        centroids_cupy_sorted = model_cupy.centroids_cpu[
            np.argsort(model_cupy.centroids_cpu[:, 0])
        ]
        centroids_bincount_sorted = model_bincount.centroids_cpu[
            np.argsort(model_bincount.centroids_cpu[:, 0])
        ]

        np.testing.assert_allclose(
            centroids_cupy_sorted,
            centroids_bincount_sorted,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Две GPU реализации дают разные центроиды",
        )

    @pytest.mark.skipif(not gpu_available(), reason="GPU недоступен")
    def test_all_implementations_consistency(self, medium_dataset):
        """Все реализации должны давать одинаковые результаты на среднем датасете."""
        X, initial_centroids = medium_dataset

        # CPU NumPy
        model_cpu = KMeansCPUNumpy(n_clusters=3, n_iters=15)
        model_cpu.fit(X, initial_centroids)

        # CPU Multiprocessing
        model_mp = KMeansCPUMultiprocessing(
            n_clusters=3,
            n_iters=15,
            mp=MultiprocessingConfig(n_processes=2),
        )
        model_mp.fit(X, initial_centroids)

        # GPU CuPy V1
        model_gpu = KMeansGPUCuPyV1(n_clusters=3, n_iters=15)
        model_gpu.fit(X, initial_centroids)

        # GPU CuPy V2 (Bincount)
        model_gpu_bc = KMeansGPUCuPyV2(n_clusters=3, n_iters=15)
        model_gpu_bc.fit(X, initial_centroids)

        # Сортируем центроиды для сравнения
        centroids_cpu_sorted = model_cpu.centroids[
            np.argsort(model_cpu.centroids[:, 0])
        ]
        centroids_mp_sorted = model_mp.centroids[np.argsort(model_mp.centroids[:, 0])]
        centroids_gpu_sorted = model_gpu.centroids_cpu[
            np.argsort(model_gpu.centroids_cpu[:, 0])
        ]
        centroids_gpu_bc_sorted = model_gpu_bc.centroids_cpu[
            np.argsort(model_gpu_bc.centroids_cpu[:, 0])
        ]

        # Все должны быть близки друг к другу
        np.testing.assert_allclose(
            centroids_cpu_sorted,
            centroids_mp_sorted,
            rtol=1e-4,
            atol=1e-4,
            err_msg="CPU NumPy и Multiprocessing не согласованы",
        )

        np.testing.assert_allclose(
            centroids_cpu_sorted,
            centroids_gpu_sorted,
            rtol=1e-4,
            atol=1e-4,
            err_msg="CPU и GPU CuPy не согласованы",
        )

        np.testing.assert_allclose(
            centroids_gpu_sorted,
            centroids_gpu_bc_sorted,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Две GPU реализации не согласованы",
        )

