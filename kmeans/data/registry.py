"""
Реестр датасетов для экспериментов.

Предоставляет интерфейс для поиска и фильтрации датасетов по метаданным
из файла datasets_summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


class DatasetRegistry:
    """
    Реестр датасетов для экспериментов.

    Загружает метаданные из datasets_summary.json и предоставляет методы
    для поиска датасетов по различным критериям.
    """

    def __init__(self, summary_path: Path | str, datasets_root: Path | str) -> None:
        """
        Инициализация реестра.

        Args:
            summary_path: Путь к файлу datasets_summary.json
            datasets_root: Корневая директория с датасетами
        """
        self.summary_path = Path(summary_path)
        self.datasets_root = Path(datasets_root)
        self.summary: dict[str, Any] = {}
        self.datasets: list[dict[str, Any]] = []

        self._load_summary()

    def _load_summary(self) -> None:
        """Загружает метаданные из datasets_summary.json."""
        with open(self.summary_path, "r", encoding="utf-8") as f:
            self.summary = json.load(f)

        self.datasets = self.summary.get("datasets", [])

    def get_all(self) -> list[dict[str, Any]]:
        """
        Возвращает список всех описаний датасетов из summary.

        Returns:
            Список словарей с метаданными датасетов.
            Каждый словарь содержит поля: N, D, K, filepath, purpose и т.д.
        """
        return list(self.datasets)

    def iter_all(self) -> Iterator[dict[str, Any]]:
        """
        Итерация по всем датасетам из реестра.

        Yields:
            Словари с ключами:
            - data_path: полный путь к файлу датасета
            - metadata: метаданные датасета
        """
        for entry in self.datasets:
            yield {
                "data_path": self.datasets_root / entry["filepath"],
                "metadata": entry,
            }

    def find(
        self,
        *,
        N: int | None = None,
        D: int | None = None,
        K: int | None = None,
        purpose: str | None = None,
    ) -> dict[str, Any]:
        """
        Находит ровно один датасет, удовлетворяющий условиям.

        Args:
            N: Количество точек (фильтр)
            D: Размерность (фильтр)
            K: Количество кластеров (фильтр)
            purpose: Назначение датасета (фильтр)

        Returns:
            Словарь с ключами:
            - data_path: полный путь к файлу датасета
            - metadata: метаданные датасета

        Raises:
            ValueError: Если датасет не найден или найдено несколько датасетов
        """
        candidates = list(self.datasets)

        # Последовательная фильтрация по критериям
        if N is not None:
            candidates = [d for d in candidates if d.get("N") == N]
        if D is not None:
            candidates = [d for d in candidates if d.get("D") == D]
        if K is not None:
            candidates = [d for d in candidates if d.get("K") == K]
        if purpose is not None:
            candidates = [
                d for d in candidates if d.get("purpose") == purpose
            ]

        if len(candidates) == 0:
            raise ValueError(
                f"Dataset not found for parameters: N={N}, D={D}, K={K}, purpose={purpose}"
            )

        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous dataset selection: found {len(candidates)} datasets "
                f"for parameters: N={N}, D={D}, K={K}, purpose={purpose}"
            )

        entry = candidates[0]

        return {
            "data_path": self.datasets_root / entry["filepath"],
            "metadata": entry,
        }
