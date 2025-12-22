# kmeans/data/registry.py

import json
from pathlib import Path

class DatasetRegistry:
    def __init__(self, summary_path: Path, datasets_root: Path):
        self.summary_path = Path(summary_path)
        self.datasets_root = Path(datasets_root)

        self._load_summary()

    def _load_summary(self):
        with open(self.summary_path, "r", encoding="utf-8") as f:
            self.summary = json.load(f)

        self.datasets = self.summary["datasets"]

    def get_all(self):
        """
        Возвращает список всех описаний датасетов из summary.

        Каждый элемент — это словарь с полями вроде
        N, D, K, filepath, purpose и т.д. (как в datasets_summary.json).
        """
        return list(self.datasets)

    def iter_all(self):
        """
        Итерация по всем датасетам из реестра.
        """
        for entry in self.datasets:
            yield {
                "data_path": self.datasets_root / entry["filepath"],
                "metadata": entry
            }


    def find(
        self,
        *,
        N=None,
        D=None,
        K=None,
        purpose=None
    ):
        """
        Возвращает ровно один датасет, удовлетворяющий условиям.
        """
        candidates = self.datasets

        if N is not None:
            candidates = [d for d in candidates if d["N"] == N]
        if D is not None:
            candidates = [d for d in candidates if d["D"] == D]
        if K is not None:
            candidates = [d for d in candidates if d["K"] == K]
        if purpose is not None:
            candidates = [d for d in candidates if d.get("purpose") == purpose]

        if len(candidates) == 0:
            raise ValueError("Dataset not found for given parameters")

        if len(candidates) > 1:
            raise ValueError("Ambiguous dataset selection")

        entry = candidates[0]

        return {
            "data_path": self.datasets_root / entry["filepath"],
            "metadata": entry
        }

