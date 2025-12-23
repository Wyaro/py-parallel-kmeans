"""
Общие фикстуры для всех тестов.
"""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def random_seed():
    """Фикстура для установки глобального seed."""
    np.random.seed(42)
    return 42


@pytest.fixture
def small_dataset():
    """Фикстура с небольшим тестовым датасетом (2D, 2 кластера)."""
    np.random.seed(42)
    # Два явно разделённых кластера
    cluster1 = np.random.randn(30, 2) + [0, 0]
    cluster2 = np.random.randn(30, 2) + [5, 5]
    X = np.vstack([cluster1, cluster2])
    initial_centroids = np.array([
        [-1.0, -1.0],
        [6.0, 6.0],
    ])
    return X, initial_centroids


@pytest.fixture
def medium_dataset():
    """Фикстура со средним тестовым датасетом (10D, 3 кластера)."""
    np.random.seed(42)
    cluster1 = np.random.randn(50, 10) + [0] * 10
    cluster2 = np.random.randn(50, 10) + [5] * 10
    cluster3 = np.random.randn(50, 10) + [-5] * 10
    X = np.vstack([cluster1, cluster2, cluster3])
    initial_centroids = np.array([
        [-1.0] * 10,
        [6.0] * 10,
        [-6.0] * 10,
    ])
    return X, initial_centroids


@pytest.fixture
def simple_2d_dataset():
    """Фикстура с очень простым 2D датасетом для базовых тестов."""
    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [10.0, 10.0],
        [11.0, 11.0],
        [12.0, 12.0],
    ])
    initial_centroids = np.array([
        [0.5, 0.5],
        [11.0, 11.0],
    ])
    return X, initial_centroids

