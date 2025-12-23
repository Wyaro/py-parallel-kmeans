"""
Анализ результатов экспериментов по производительности K-means.

Модуль читает результаты таймингов из JSON/NDJSON файла и выводит
статистику по времени выполнения различных этапов алгоритма.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


class TimingResultsAnalyzer:
    """
    Анализатор результатов экспериментов по производительности.

    Читает результаты из файла и вычисляет статистику по времени выполнения.
    """

    def __init__(self, json_path: Path | str, n_iters: int = 100) -> None:
        """
        Инициализация анализатора.

        Args:
            json_path: Путь к файлу с результатами (JSON или NDJSON)
            n_iters: Количество итераций алгоритма (для нормализации времени)
        """
        self.json_path = Path(json_path)
        self.n_iters = n_iters

        if not self.json_path.exists():
            raise FileNotFoundError(
                f"Timing results file not found: {self.json_path}"
            )

    def _iter_results(self) -> Iterable[dict[str, Any]]:
        """
        Итерирует результаты таймингов из файла.

        Поддерживает два формата:
        - JSON-массив (старый формат): [{"experiment": ...}, ...]
        - NDJSON (новый формат): каждая строка — отдельный JSON-объект

        Yields:
            Словари с результатами экспериментов
        """
        with self.json_path.open("r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)

            if first_char == "[":
                # Старый формат: JSON-массив
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(
                        "Expected top-level JSON array in timing results file"
                    )
                for entry in data:
                    if isinstance(entry, dict):
                        yield entry
            else:
                # Новый формат: NDJSON (каждая строка — JSON)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(entry, dict):
                        yield entry

    def _compute_statistics(
        self, runs: list[dict[str, Any]]
    ) -> dict[str, float]:
        """
        Вычисляет статистику по результатам прогонов.

        Args:
            runs: Список словарей с результатами отдельных прогонов

        Returns:
            Словарь со статистикой (средние, медианы в миллисекундах)
        """
        # Преобразуем время из секунд в миллисекунды
        # Для T_assign, T_update, T_iter — это время всех итераций,
        # поэтому делим на n_iters для получения времени одной итерации
        t_assign_ms = [
            float(r["T_assign_total"]) * 1000.0 / float(self.n_iters)
            for r in runs
        ]
        t_update_ms = [
            float(r["T_update_total"]) * 1000.0 / float(self.n_iters)
            for r in runs
        ]
        t_iter_ms = [
            float(r["T_iter_total"]) * 1000.0 / float(self.n_iters) for r in runs
        ]
        # T_fit — это общее время одного запуска алгоритма
        t_total_ms = [float(r["T_fit"]) * 1000.0 for r in runs]

        return {
            "assign_mean": mean(t_assign_ms),
            "assign_med": median(t_assign_ms),
            "update_mean": mean(t_update_ms),
            "update_med": median(t_update_ms),
            "iter_mean": mean(t_iter_ms),
            "iter_med": median(t_iter_ms),
            "total_mean": mean(t_total_ms),
            "total_med": median(t_total_ms),
        }

    def analyze(self) -> None:
        """
        Анализирует результаты и выводит статистику в консоль.

        Для каждого эксперимента выводит:
        - Средние и медианные значения времени назначения кластеров
        - Средние и медианные значения времени обновления центроидов
        - Средние и медианные значения времени одной итерации
        - Средние и медианные значения общего времени выполнения
        """
        for entry in self._iter_results():
            exp = entry.get("experiment", "unknown")
            impl = entry.get("implementation", "unknown")
            dataset_info: dict[str, Any] = entry.get("dataset", {}) or {}
            timing: dict[str, Any] = entry.get("timing", {}) or {}

            runs = timing.get("runs") or []
            if not runs:
                continue

            stats = self._compute_statistics(runs)

            print(f"Эксперимент: {exp}, реализация: {impl}")
            print(
                f"  Датасет: N={dataset_info.get('N')}, "
                f"D={dataset_info.get('D')}, "
                f"K={dataset_info.get('K')}"
            )
            print(
                f"  Tназначения (одна итерация): "
                f"ср={stats['assign_mean']:.3f} мс, "
                f"мед={stats['assign_med']:.3f} мс"
            )
            print(
                f"  Tобновления (одна итерация): "
                f"ср={stats['update_mean']:.3f} мс, "
                f"мед={stats['update_med']:.3f} мс"
            )
            print(
                f"  Tитерации (одна итерация): "
                f"ср={stats['iter_mean']:.3f} мс, "
                f"мед={stats['iter_med']:.3f} мс"
            )
            print(
                f"  Tобщ (один запуск алгоритма): "
                f"ср={stats['total_mean']:.3f} мс, "
                f"мед={stats['total_med']:.3f} мс"
            )
            print()


def compute_stats_from_results(
    json_path: str | Path, n_iters: int = 100
) -> None:
    """
    Удобная функция для быстрого анализа результатов.

    Args:
        json_path: Путь к файлу с результатами
        n_iters: Количество итераций алгоритма
    """
    analyzer = TimingResultsAnalyzer(json_path, n_iters)
    analyzer.analyze()
