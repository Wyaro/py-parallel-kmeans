import numpy as np
import json
from pathlib import Path
import logging

class Dataset:
    def __init__(self, data_dir: str, dataset_info: dict):
        """
        data_dir: путь к папке с датасетами
        dataset_info: словарь с полями из datasets_summary.json для конкретного датасета
        """
        self.data_dir = Path(data_dir)
        self.dataset_info = dataset_info

        self.data_path = self.data_dir / dataset_info["filepath"]
        self.X = None
        self.labels_true = None
        self.initial_centroids = None

        logging.info(f"Loading dataset from {self.data_path}")
        self._load_data()

    def _load_data(self):
        K = self.dataset_info["K"]
        centroids = []
        points = []
        labels = []

        with open(self.data_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                label = int(parts[0])
                values = np.array(parts[1:], dtype=np.float64)

                if len(centroids) < K:
                    centroids.append(values)
                else:
                    points.append(values)
                    labels.append(label)

        self.initial_centroids = np.vstack(centroids)
        self.X = np.vstack(points)
        self.labels_true = np.array(labels, dtype=np.int32)
        logging.info(f"Dataset loaded: X.shape={self.X.shape}, initial_centroids.shape={self.initial_centroids.shape}")
