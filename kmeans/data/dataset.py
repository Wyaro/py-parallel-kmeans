"""
Загрузка и представление датасетов для экспериментов K-means.

Модуль предоставляет класс Dataset для загрузки данных из текстовых файлов
в формате, созданном DatasetGenerator.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np


class Dataset:
    """
    Представление датасета для экспериментов K-means.

    Загружает данные из текстового файла в формате:
    # Метаданные в JSON
    # Центроиды (K строк: метка + координаты)
    # Данные (N строк: метка + координаты)
    """

    def __init__(self, data_dir: str | Path, dataset_info: dict[str, Any]) -> None:
        """
        Инициализация датасета.

        Args:
            data_dir: Путь к корневой директории с датасетами
            dataset_info: Словарь с метаданными датасета из datasets_summary.json
                Должен содержать ключ "filepath" с относительным путём к файлу
        """
        self.data_dir = Path(data_dir)
        self.dataset_info = dataset_info

        self.data_path = self.data_dir / dataset_info["filepath"]
        self.X: np.ndarray | None = None
        self.labels_true: np.ndarray | None = None
        self.initial_centroids: np.ndarray | None = None

        logging.info(f"Loading dataset from {self.data_path}")
        self._load_data()

    def _load_data(self) -> None:
        """
        Загружает данные из текстового файла.

        Формат файла:
        - Первая строка: JSON с метаданными (начинается с #)
        - Следующие K строк: центроиды (метка + координаты)
        - Пустая строка
        - Следующие N строк: точки данных (метка + координаты)
        """
        K = self.dataset_info["K"]
        centroids: list[np.ndarray] = []
        points: list[np.ndarray] = []
        labels: list[int] = []

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Пропускаем комментарии и пустые строки
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if not parts:
                    continue

                label = int(parts[0])
                values = np.array(parts[1:], dtype=np.float64)

                # Первые K строк с метками 0..K-1 — это центроиды
                if len(centroids) < K and label < K:
                    centroids.append(values)
                else:
                    points.append(values)
                    labels.append(label)

        # Преобразуем в numpy массивы
        self.initial_centroids = np.vstack(centroids)
        self.X = np.vstack(points)
        self.labels_true = np.array(labels, dtype=np.int32)

        logging.info(
            f"Dataset loaded: X.shape={self.X.shape}, "
            f"initial_centroids.shape={self.initial_centroids.shape}"
        )
