"""
Генератор синтетических датасетов для бенчмаркинга K-means.

Создаёт наборы данных с различными параметрами N, D, K для экспериментов
по масштабированию производительности алгоритма кластеризации.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetConfig:
    """Конфигурация параметров датасета."""

    N: int
    D: int
    K: int
    cluster_std: float = 1.0
    seed_offset: int = 0
    purpose: str | None = None


@dataclass
class GeneratedDataset:
    """Контейнер для сгенерированных данных."""

    data: np.ndarray
    labels: np.ndarray
    centers: np.ndarray
    metadata: dict[str, Any]


class DatasetGenerator:
    """
    Генератор синтетических датасетов для бенчмаркинга.

    Использует sklearn.make_blobs для создания кластеризованных данных
    с заданными параметрами и сохраняет их в текстовом формате.
    """

    # Директории для различных типов датасетов
    DIRECTORIES = ["base", "scaling_N", "scaling_D", "scaling_K", "validation"]

    def __init__(self, base_seed: int = 42) -> None:
        """
        Инициализация генератора.

        Args:
            base_seed: Базовое значение seed для воспроизводимости
        """
        self.base_seed = base_seed
        self.datasets_dir = Path("datasets")
        self.create_directory_structure()

    def create_directory_structure(self) -> None:
        """Создаёт необходимую структуру директорий для датасетов."""
        for dir_name in self.DIRECTORIES:
            (self.datasets_dir / dir_name).mkdir(parents=True, exist_ok=True)

    def generate_blobs_dataset(
        self,
        N: int,
        D: int,
        K: int,
        cluster_std: float = 1.0,
        seed_offset: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Генерация синтетического датасета с помощью make_blobs.

        Args:
            N: Количество точек
            D: Размерность пространства
            K: Количество кластеров
            cluster_std: Стандартное отклонение кластеров
            seed_offset: Смещение seed для разных конфигураций

        Returns:
            Кортеж (data, labels, centers):
            - data: массив данных (N x D)
            - labels: метки кластеров (N,)
            - centers: центры кластеров (K x D)
        """
        np.random.seed(self.base_seed)

        print(f"Генерация: N={N:,}, D={D}, K={K}, seed={self.base_seed}")

        data, labels, centers = make_blobs(
            n_samples=N,
            n_features=D,
            centers=K,
            cluster_std=cluster_std,
            center_box=(-10.0, 10.0),
            random_state=self.base_seed,
            return_centers=True,
        )

        # Нормализация данных
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        centers = scaler.transform(centers)

        return data, labels, centers

    def _create_metadata(
        self, config: DatasetConfig, data: np.ndarray, centers: np.ndarray
    ) -> dict[str, Any]:
        """
        Создаёт метаданные для датасета.

        Args:
            config: Конфигурация датасета
            data: Массив данных
            centers: Центры кластеров

        Returns:
            Словарь метаданных
        """
        return {
            "N": config.N,
            "D": config.D,
            "K": config.K,
            "cluster_std": config.cluster_std,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_points": len(data),
            "dimensions": data.shape[1],
            "n_clusters": len(centers),
            "data_type": "synthetic_blobs",
            "normalized": True,
            "seed": self.base_seed,
            **({"purpose": config.purpose} if config.purpose else {}),
        }

    def save_dataset_txt(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray,
        filepath: Path,
        metadata: dict[str, Any],
    ) -> None:
        """
        Сохранение датасета в текстовом формате.

        Формат файла:
        # Метаданные в формате JSON
        # Центроиды (K строк, D+1 колонок: метка + координаты)
        # Данные (N строк, D+1 колонок: метка + координаты)

        Args:
            data: Массив данных (N x D)
            labels: Метки кластеров (N,)
            centers: Центры кластеров (K x D)
            filepath: Путь к файлу для сохранения
            metadata: Метаданные датасета
        """
        filepath = Path(filepath)

        # Запись основного файла
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# " + json.dumps(metadata, ensure_ascii=False) + "\n")
            f.write("\n")

            # Запись центроидов
            f.write("# Centroids (label, x1, x2, ..., xD)\n")
            for k in range(len(centers)):
                centroid_line = f"{k} " + " ".join(
                    [f"{coord:.8f}" for coord in centers[k]]
                )
                f.write(centroid_line + "\n")
            f.write("\n")

            # Запись точек данных
            f.write("# Data points (label, x1, x2, ..., xD)\n")
            for i in range(len(data)):
                point_line = (
                    f"{labels[i]} " + " ".join([f"{coord:.8f}" for coord in data[i]])
                )
                f.write(point_line + "\n")

        # Обновление метаданных в отдельном файле
        metadata_path = filepath.parent / "metadata.json"
        existing_metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                existing_metadata = json.load(f)

        existing_metadata[filepath.stem] = metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(existing_metadata, f, indent=2, ensure_ascii=False)

        print(
            f"  Сохранено: {filepath} ({data.shape[0]:,} точек, {data.shape[1]}D)"
        )

    def _generate_and_save(
        self,
        config: DatasetConfig,
        subdirectory: str,
        filename: str | None = None,
    ) -> None:
        """
        Вспомогательный метод: генерирует и сохраняет датасет.

        Args:
            config: Конфигурация датасета
            subdirectory: Поддиректория для сохранения
            filename: Имя файла (если None, генерируется автоматически)
        """
        data, labels, centers = self.generate_blobs_dataset(
            N=config.N,
            D=config.D,
            K=config.K,
            cluster_std=config.cluster_std,
            seed_offset=config.seed_offset,
        )

        metadata = self._create_metadata(config, data, centers)

        if filename is None:
            filename = f"N{config.N}_D{config.D}_K{config.K}.txt"

        filepath = self.datasets_dir / subdirectory / filename
        self.save_dataset_txt(data, labels, centers, filepath, metadata)

    def generate_base_configurations(self) -> None:
        """Генерация базовых конфигураций для экспериментов."""
        print("\n" + "=" * 60)
        print("Генерация базовых конфигураций")
        print("=" * 60)

        config = DatasetConfig(N=100_000, D=50, K=8, cluster_std=1.0)
        self._generate_and_save(config, "base")

    def generate_scaling_N(self) -> None:
        """Генерация датасетов для масштабирования по количеству точек."""
        print("\n" + "=" * 60)
        print("Генерация датасетов: масштабирование по N")
        print("=" * 60)

        N_values = [1_000, 100_000, 1_000_000, 1_000_000_000]

        for i, N in enumerate(N_values):
            # Для больших N используем экономную генерацию
            if N > 10_000_000:
                print(
                    f"ВНИМАНИЕ: N={N:,} слишком велико для памяти. "
                    f"Генерируем уменьшенную версию."
                )
                N = 5_000_000  # Практический предел для большинства систем

            config = DatasetConfig(
                N=N,
                D=50,
                K=8,
                cluster_std=1.0,
                seed_offset=i,
                purpose="scaling_by_N",
            )
            self._generate_and_save(config, "scaling_N")

    def generate_scaling_D(self) -> None:
        """Генерация датасетов для масштабирования по размерности."""
        print("\n" + "=" * 60)
        print("Генерация датасетов: масштабирование по D")
        print("=" * 60)

        D_values = [2, 10, 50, 200]

        for i, D in enumerate(D_values):
            config = DatasetConfig(
                N=100_000,
                D=D,
                K=8,
                cluster_std=1.0,
                seed_offset=i + 10,
                purpose="scaling_by_D",
            )
            self._generate_and_save(config, "scaling_D")

    def generate_scaling_K(self) -> None:
        """Генерация датасетов для масштабирования по количеству кластеров."""
        print("\n" + "=" * 60)
        print("Генерация датасетов: масштабирование по K")
        print("=" * 60)

        K_values = [4, 8, 16, 32]

        for i, K in enumerate(K_values):
            config = DatasetConfig(
                N=100_000,
                D=50,
                K=K,
                cluster_std=1.0,
                seed_offset=i + 20,
                purpose="scaling_by_K",
            )
            self._generate_and_save(config, "scaling_K")

    def generate_validation_data(self) -> None:
        """Генерация небольшого датасета для валидации корректности."""
        print("\n" + "=" * 60)
        print("Подготовка данных для валидации")
        print("=" * 60)

        config = DatasetConfig(
            N=1_000,
            D=10,
            K=4,
            cluster_std=0.5,
            seed_offset=100,
            purpose="validation",
        )
        self._generate_and_save(config, "validation", "validation_dataset.txt")

    def create_summary(self) -> None:
        """Создаёт сводный файл datasets_summary.json со всеми датасетами."""
        summary: dict[str, Any] = {
            "project": "K-means Benchmark Datasets",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_seed": self.base_seed,
            "total_datasets": 0,
            "datasets": [],
        }

        # Собираем информацию о всех датасетах
        for root, _, files in os.walk(self.datasets_dir):
            for file in files:
                if file.endswith(".txt") and not file.startswith("."):
                    filepath = Path(root) / file
                    rel_path = filepath.relative_to(self.datasets_dir)

                    # Читаем первую строку с метаданными
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            first_line = f.readline().strip("# \n")
                            metadata = json.loads(first_line)
                            metadata["filepath"] = str(rel_path)
                            metadata["size_mb"] = os.path.getsize(filepath) / (
                                1024 * 1024
                            )
                            summary["datasets"].append(metadata)
                            summary["total_datasets"] += 1
                    except (json.JSONDecodeError, KeyError, ValueError):
                        # Пропускаем файлы с некорректными метаданными
                        continue

        # Сохраняем summary
        summary_path = self.datasets_dir / "datasets_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"Сводный отчет: {summary_path}")

        # Выводим статистику
        print("\nСтатистика датасетов:")
        categories: dict[str, int] = {}
        for ds in summary["datasets"]:
            cat = ds.get("purpose", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} датасетов")

    def generate_all(self) -> None:
        """Генерирует все типы датасетов и создаёт сводный файл."""
        print("Начало генерации синтетических датасетов")
        print(f"Базовый seed: {self.base_seed}")
        print(f"Время начала: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        start_time = time.time()

        self.generate_base_configurations()
        self.generate_scaling_N()
        self.generate_scaling_D()
        self.generate_scaling_K()
        self.generate_validation_data()

        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Генерация завершена за {elapsed:.2f} секунд")
        print(f"Датасеты сохранены в: {self.datasets_dir.absolute()}")

        # Создание summary
        self.create_summary()


if __name__ == "__main__":
    generator = DatasetGenerator(base_seed=42)
    generator.generate_all()
