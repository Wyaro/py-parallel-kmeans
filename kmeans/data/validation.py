"""
Валидация датасетов для проверки корректности загрузки.

Модуль предоставляет функции для проверки соответствия загруженных данных
их метаданным.
"""

from __future__ import annotations

from kmeans.data.dataset import Dataset


def validate_dataset(dataset: Dataset) -> None:
    """
    Проверяет корректность загруженного датасета.

    Проверяет соответствие размеров данных метаданным из dataset_info.

    Args:
        dataset: Экземпляр Dataset для валидации

    Raises:
        AssertionError: Если размеры данных не соответствуют метаданным
    """
    meta = dataset.dataset_info

    assert dataset.X is not None, "Dataset data (X) is None"
    assert (
        dataset.initial_centroids is not None
    ), "Initial centroids are None"

    assert dataset.X.shape[0] == meta["N"], (
        f"Expected {meta['N']} points, got {dataset.X.shape[0]}"
    )
    assert dataset.X.shape[1] == meta["D"], (
        f"Expected {meta['D']} dimensions, got {dataset.X.shape[1]}"
    )
    assert dataset.initial_centroids.shape == (
        meta["K"],
        meta["D"],
    ), (
        f"Expected centroids shape ({meta['K']}, {meta['D']}), "
        f"got {dataset.initial_centroids.shape}"
    )
