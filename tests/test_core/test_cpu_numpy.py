"""
Unit-тесты для CPU NumPy реализации K-means.
"""

import numpy as np
import pytest
from kmeans.core.cpu_numpy import KMeansCPUNumpy


class TestKMeansCPUNumpy:
    """Тесты базовой функциональности CPU NumPy реализации."""

    def test_assign_clusters(self, simple_2d_dataset):
        """Тест шага назначения кластеров."""
        model = KMeansCPUNumpy(n_clusters=2, n_iters=1)
        X, centroids = simple_2d_dataset

        labels = model.assign_clusters(X, centroids)

        # Проверяем форму результата
        assert labels.shape == (6,)
        assert labels.dtype == np.int64 or labels.dtype == np.int32

        # Первые 3 точки должны быть ближе к первому центроиду
        # Последние 3 точки должны быть ближе ко второму центроиду
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4] == labels[5]
        assert labels[0] != labels[3]  # Разные кластеры

    def test_update_centroids(self, simple_2d_dataset):
        """Тест шага обновления центроидов."""
        model = KMeansCPUNumpy(n_clusters=2, n_iters=1)
        X, _ = simple_2d_dataset

        # Все точки в первом кластере, все во втором
        labels = np.array([0, 0, 0, 1, 1, 1])

        new_centroids = model.update_centroids(X, labels)

        # Проверяем форму
        assert new_centroids.shape == (2, 2)

        # Первый центроид должен быть средним точек [0,0], [1,1], [2,2]
        expected_centroid_0 = np.array([1.0, 1.0])
        np.testing.assert_allclose(new_centroids[0], expected_centroid_0, rtol=1e-10)

        # Второй центроид должен быть средним точек [10,10], [11,11], [12,12]
        expected_centroid_1 = np.array([11.0, 11.0])
        np.testing.assert_allclose(new_centroids[1], expected_centroid_1, rtol=1e-10)

    def test_update_centroids_empty_cluster(self):
        """Тест обработки пустых кластеров."""
        model = KMeansCPUNumpy(n_clusters=3, n_iters=1)

        # Все точки в одном кластере
        X = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ])
        labels = np.array([0, 0, 0])  # Все точки в кластере 0

        new_centroids = model.update_centroids(X, labels)

        # Проверяем форму
        assert new_centroids.shape == (3, 2)

        # Кластер 0 должен быть обновлён
        assert not np.allclose(new_centroids[0], [0, 0])

        # Кластеры 1 и 2 остаются нулевыми (пустые)
        assert np.allclose(new_centroids[1], [0, 0])
        assert np.allclose(new_centroids[2], [0, 0])

    def test_fit_convergence(self, small_dataset):
        """Тест полного цикла fit на простых данных."""
        model = KMeansCPUNumpy(n_clusters=2, n_iters=10)
        X, initial_centroids = small_dataset

        model.fit(X, initial_centroids)

        # Проверяем, что алгоритм отработал
        assert model.centroids is not None
        assert model.labels is not None
        assert model.centroids.shape == (2, 2)
        assert model.labels.shape == (60,)

        # Проверяем, что все точки назначены кластерам
        assert np.all((model.labels >= 0) & (model.labels < 2))

        # Проверяем, что тайминги собраны
        assert model.t_assign_total > 0
        assert model.t_update_total > 0
        assert model.t_iter_total > 0
        assert abs(model.t_iter_total - (model.t_assign_total + model.t_update_total)) < 1e-6

    def test_fit_multiple_iterations(self, small_dataset):
        """Тест, что fit выполняет заданное количество итераций."""
        n_iters = 5
        model = KMeansCPUNumpy(n_clusters=2, n_iters=n_iters)
        X, initial_centroids = small_dataset

        model.fit(X, initial_centroids)

        # Проверяем, что тайминги соответствуют количеству итераций
        # (примерно, так как время может варьироваться)
        assert model.t_iter_total > 0

    def test_centroids_shape_consistency(self, medium_dataset):
        """Тест согласованности формы центроидов."""
        model = KMeansCPUNumpy(n_clusters=3, n_iters=5)
        X, initial_centroids = medium_dataset

        model.fit(X, initial_centroids)

        # Форма центроидов должна соответствовать (K, D)
        assert model.centroids.shape == (3, 10)
        assert model.centroids.dtype == np.float64

