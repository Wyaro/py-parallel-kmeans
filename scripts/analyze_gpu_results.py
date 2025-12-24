"""
Анализ результатов GPU-реализаций K-means для выявления проблем производительности.

Модуль читает файл с результатами экспериментов (JSON-массив или NDJSON),
выделяет GPU-реализации и сравнивает их по времени одной итерации и общему
времени выполнения, используя базовую версию `python_gpu_cupy_v1` как эталон.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import argparse


GPU_IMPLEMENTATIONS: tuple[str, ...] = (
    "python_gpu_cupy_v1",
    "python_gpu_cupy_v2",
    "python_gpu_cupy_v3",
    "python_gpu_cupy_v4",
)


def _read_entries(json_path: Path) -> List[Dict[str, Any]]:
    """Читает файл с результатами и возвращает список JSON-объектов."""
    with json_path.open("r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            data = json.load(f)
            return data if isinstance(data, list) else []

        entries: list[dict[str, Any]] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                entries.append(entry)
        return entries


def _group_gpu_results(
    entries: Iterable[Mapping[str, Any]],
) -> Dict[Tuple[Any, Any, Any], Dict[str, Dict[str, float]]]:
    """
    Группирует результаты GPU-реализаций по датасетам.

    Returns:
        Словарь: (N, D, K) -> { implementation_name -> агрегированные метрики }.
    """
    results_by_dataset: dict[tuple[Any, Any, Any], dict[str, dict[str, float]]] = {}

    for entry in entries:
        impl = entry.get("implementation", "")
        if impl not in GPU_IMPLEMENTATIONS:
            continue

        dataset_info = entry.get("dataset", {}) or {}
        timing = entry.get("timing", {}) or {}

        key = (
            dataset_info.get("N"),
            dataset_info.get("D"),
            dataset_info.get("K"),
        )
        if key not in results_by_dataset:
            results_by_dataset[key] = {}

        runs = timing.get("runs") or []
        if not runs:
            continue

        # Вычисляем средние значения времени одной итерации
        t_assign_per_iter: list[float] = []
        t_update_per_iter: list[float] = []
        t_fit_total: list[float] = []

        for r in runs:
            n_iters = float(r.get("n_iters_actual", 2))
            if n_iters > 0:
                t_assign_per_iter.append(float(r["T_assign_total"]) / n_iters)
                t_update_per_iter.append(float(r["T_update_total"]) / n_iters)
            t_fit_total.append(float(r["T_fit"]))

        results_by_dataset[key][impl] = {
            "T_assign_per_iter_mean": mean(t_assign_per_iter)
            if t_assign_per_iter
            else 0.0,
            "T_assign_per_iter_med": median(t_assign_per_iter)
            if t_assign_per_iter
            else 0.0,
            "T_update_per_iter_mean": mean(t_update_per_iter)
            if t_update_per_iter
            else 0.0,
            "T_update_per_iter_med": median(t_update_per_iter)
            if t_update_per_iter
            else 0.0,
            "T_fit_mean": mean(t_fit_total) if t_fit_total else 0.0,
            "T_fit_med": median(t_fit_total) if t_fit_total else 0.0,
            "T_transfer_avg": float(timing.get("T_transfer_avg", 0.0)),
            "T_transfer_ratio_avg": float(timing.get("T_transfer_ratio_avg", 0.0)),
        }

    return results_by_dataset


def analyze_gpu_results(json_path: str | Path) -> None:
    """Анализирует результаты GPU-реализаций и печатает сводку в stdout."""

    json_path = Path(json_path)

    entries = _read_entries(json_path)
    results_by_dataset = _group_gpu_results(entries)

    # Вывод результатов
    print("=" * 80)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ GPU РЕАЛИЗАЦИЙ")
    print("=" * 80)
    print()

    for (N, D, K), impls in sorted(results_by_dataset.items()):
        print(f"Датасет: N={N}, D={D}, K={K}")
        print("-" * 80)

        # Базовая реализация как эталон
        baseline = impls.get("python_gpu_cupy_v1")
        if not baseline:
            print("  ⚠ Базовая реализация (python_gpu_cupy_v1) отсутствует!")
            continue

        print(
            f"\n{'Реализация':<30} "
            f"{'T_fit (мс)':<15} "
            f"{'T_assign/iter (мс)':<20} "
            f"{'T_update/iter (мс)':<20} "
            f"{'T_transfer %':<15}"
        )
        print("-" * 80)

        for impl_name in GPU_IMPLEMENTATIONS:
            if impl_name not in impls:
                continue

            data = impls[impl_name]
            t_fit_ms = data["T_fit_mean"] * 1000.0
            t_assign_ms = data["T_assign_per_iter_mean"] * 1000.0
            t_update_ms = data["T_update_per_iter_mean"] * 1000.0
            transfer_ratio = data["T_transfer_ratio_avg"]

            # Сравнение с базовой версией
            baseline_fit = baseline["T_fit_mean"]
            speedup = baseline_fit / data["T_fit_mean"] if data["T_fit_mean"] > 0 else 0.0
            speedup_str = f"({speedup:.2f}x)" if speedup != 1.0 else ""

            print(
                f"{impl_name:<30} "
                f"{t_fit_ms:>10.3f} {speedup_str:<5} "
                f"{t_assign_ms:>15.3f} "
                f"{t_update_ms:>15.3f} "
                f"{transfer_ratio:>10.2f}%"
            )

        print()

        # Анализ проблем
        print("  Анализ:")
        for impl_name in GPU_IMPLEMENTATIONS:
            if impl_name == "python_gpu_cupy_v1" or impl_name not in impls:
                continue

            data = impls[impl_name]
            baseline_t_fit = baseline["T_fit_mean"]
            impl_t_fit = data["T_fit_mean"]

            if impl_t_fit > baseline_t_fit * 1.1:  # Более чем на 10% медленнее
                slowdown = impl_t_fit / baseline_t_fit
                print(
                    f"    ⚠ {impl_name}: в {slowdown:.2f} раз медленнее базовой версии"
                )

                # Анализ причин
                baseline_t_update = baseline["T_update_per_iter_mean"]
                impl_t_update = data["T_update_per_iter_mean"]

                if impl_t_update > baseline_t_update * 1.5:
                    print(
                        "       → Проблема в update_centroids: "
                        f"{impl_t_update*1000:.3f} мс vs {baseline_t_update*1000:.3f} мс"
                    )
                    if "bincount" in impl_name:
                        print(
                            "       → Возможная причина: цикл по D создаёт "
                            "накладные расходы на маленьких датасетах"
                        )

        print("\n" + "=" * 80 + "\n")


def _cli() -> None:
    """CLI-обёртка для анализа GPU-реализаций."""
    parser = argparse.ArgumentParser(
        description=(
            "Анализ и сравнение GPU-реализаций K-means "
            "на основе файла с таймингами."
        )
    )
    parser.add_argument(
        "json_path",
        nargs="?",
        type=Path,
        default=Path("kmeans_timing_results.json"),
        help=(
            "Путь к файлу с результатами таймингов "
            "(по умолчанию: ./kmeans_timing_results.json)"
        ),
    )

    args = parser.parse_args()
    analyze_gpu_results(args.json_path)


if __name__ == "__main__":
    _cli()