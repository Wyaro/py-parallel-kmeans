from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


def _iter_results(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array in timing results file")
    for entry in data:
        if isinstance(entry, dict):
            yield entry


def compute_stats_from_results(json_path: str | Path, n_iters: int = 100) -> None:
    """
    Читает kmeans_timing_results.json и печатает табличку со средними и
    медианными значениями (в миллисекундах) для:
    - Tназначения (T_assign_total)
    - Tобновления (T_update_total)
    - Tитерации (T_iter_total)
    - Tобщего времени (T_fit)
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Timing results file not found: {json_path}")

    for entry in _iter_results(json_path):
        exp = entry.get("experiment")
        impl = entry.get("implementation")
        dataset_info: dict[str, Any] = entry.get("dataset", {}) or {}
        timing: dict[str, Any] = entry.get("timing", {}) or {}

        runs = timing.get("runs") or []
        if not runs:
            continue

        # собираем значения по всем повторам (секунды → миллисекунды)
        # В логике отчёта:
        # - Tназначения, Tобновления, Tитер — ВРЕМЯ ОДНОЙ ИТЕРАЦИИ (per-iteration),
        #   поэтому берём сумму по итерациям и делим на n_iters;
        # - Tобщ — общее время одного запуска алгоритма (fit), берём T_fit.
        t_assign_ms = [
            float(r["T_assign_total"]) * 1000.0 / float(n_iters) for r in runs
        ]
        t_update_ms = [
            float(r["T_update_total"]) * 1000.0 / float(n_iters) for r in runs
        ]
        t_iter_ms = [
            float(r["T_iter_total"]) * 1000.0 / float(n_iters) for r in runs
        ]
        t_total_ms = [float(r["T_fit"]) * 1000.0 for r in runs]

        assign_mean = mean(t_assign_ms)
        assign_med = median(t_assign_ms)
        update_mean = mean(t_update_ms)
        update_med = median(t_update_ms)
        iter_mean = mean(t_iter_ms)
        iter_med = median(t_iter_ms)
        total_mean = mean(t_total_ms)
        total_med = median(t_total_ms)

        print(f"Эксперимент: {exp}, реализация: {impl}")
        print(
            f"  Датасет: N={dataset_info.get('N')},"
            f" D={dataset_info.get('D')},"
            f" K={dataset_info.get('K')}"
        )
        print(
            f"  Tназначения (одна итерация): ср={assign_mean:.3f} мс,"
            f" мед={assign_med:.3f} мс"
        )
        print(
            f"  Tобновления (одна итерация): ср={update_mean:.3f} мс,"
            f" мед={update_med:.3f} мс"
        )
        print(
            f"  Tитерации (одна итерация): ср={iter_mean:.3f} мс,"
            f" мед={iter_med:.3f} мс"
        )
        print(
            f"  Tобщ (один запуск алгоритма): ср={total_mean:.3f} мс,"
            f" мед={total_med:.3f} мс"
        )
        print()







