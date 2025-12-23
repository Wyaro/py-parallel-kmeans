"""
Анализ результатов GPU реализаций для выявления проблем производительности.
"""

import json
from pathlib import Path
from statistics import mean, median
from typing import Any


def analyze_gpu_results(json_path: str | Path) -> None:
    """Анализирует результаты GPU реализаций и выводит сравнение."""
    
    json_path = Path(json_path)
    gpu_implementations = [
        "python_gpu_cupy",
        "python_gpu_cupy_bincount", 
        "python_gpu_cupy_fast",
        "python_gpu_cupy_raw"
    ]
    
    results_by_dataset = {}
    
    # Чтение данных
    with json_path.open("r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == "[":
            data = json.load(f)
            entries = data if isinstance(data, list) else []
        else:
            entries = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    # Группировка по датасетам
    for entry in entries:
        impl = entry.get("implementation", "")
        if impl not in gpu_implementations:
            continue
            
        dataset_info = entry.get("dataset", {})
        timing = entry.get("timing", {})
        
        key = (dataset_info.get("N"), dataset_info.get("D"), dataset_info.get("K"))
        if key not in results_by_dataset:
            results_by_dataset[key] = {}
        
        runs = timing.get("runs", [])
        if not runs:
            continue
        
        # Вычисляем средние значения времени одной итерации
        t_assign_per_iter = []
        t_update_per_iter = []
        t_fit_total = []
        
        for r in runs:
            n_iters = float(r.get("n_iters_actual", 2))
            if n_iters > 0:
                t_assign_per_iter.append(float(r["T_assign_total"]) / n_iters)
                t_update_per_iter.append(float(r["T_update_total"]) / n_iters)
            t_fit_total.append(float(r["T_fit"]))
        
        results_by_dataset[key][impl] = {
            "T_assign_per_iter_mean": mean(t_assign_per_iter) if t_assign_per_iter else 0,
            "T_assign_per_iter_med": median(t_assign_per_iter) if t_assign_per_iter else 0,
            "T_update_per_iter_mean": mean(t_update_per_iter) if t_update_per_iter else 0,
            "T_update_per_iter_med": median(t_update_per_iter) if t_update_per_iter else 0,
            "T_fit_mean": mean(t_fit_total) if t_fit_total else 0,
            "T_fit_med": median(t_fit_total) if t_fit_total else 0,
            "T_transfer_avg": timing.get("T_transfer_avg", 0),
            "T_transfer_ratio_avg": timing.get("T_transfer_ratio_avg", 0),
        }
    
    # Вывод результатов
    print("=" * 80)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ GPU РЕАЛИЗАЦИЙ")
    print("=" * 80)
    print()
    
    for (N, D, K), impls in sorted(results_by_dataset.items()):
        print(f"Датасет: N={N}, D={D}, K={K}")
        print("-" * 80)
        
        # Базовая реализация как эталон
        baseline = impls.get("python_gpu_cupy")
        if not baseline:
            print("  ⚠ Базовая реализация отсутствует!")
            continue
        
        print(f"\n{'Реализация':<30} {'T_fit (мс)':<15} {'T_assign/iter (мс)':<20} {'T_update/iter (мс)':<20} {'T_transfer %':<15}")
        print("-" * 80)
        
        for impl_name in gpu_implementations:
            if impl_name not in impls:
                continue
                
            data = impls[impl_name]
            t_fit_ms = data["T_fit_mean"] * 1000
            t_assign_ms = data["T_assign_per_iter_mean"] * 1000
            t_update_ms = data["T_update_per_iter_mean"] * 1000
            transfer_ratio = data["T_transfer_ratio_avg"]
            
            # Сравнение с базовой версией
            speedup = baseline["T_fit_mean"] / data["T_fit_mean"] if data["T_fit_mean"] > 0 else 0
            speedup_str = f"({speedup:.2f}x)" if speedup != 1.0 else ""
            
            print(f"{impl_name:<30} {t_fit_ms:>10.3f} {speedup_str:<5} {t_assign_ms:>15.3f} {t_update_ms:>15.3f} {transfer_ratio:>10.2f}%")
        
        print()
        
        # Анализ проблем
        print("  Анализ:")
        for impl_name in gpu_implementations:
            if impl_name == "python_gpu_cupy" or impl_name not in impls:
                continue
            
            data = impls[impl_name]
            baseline_t_fit = baseline["T_fit_mean"]
            impl_t_fit = data["T_fit_mean"]
            
            if impl_t_fit > baseline_t_fit * 1.1:  # Более чем на 10% медленнее
                slowdown = impl_t_fit / baseline_t_fit
                print(f"    ⚠ {impl_name}: в {slowdown:.2f} раз медленнее базовой версии")
                
                # Анализ причин
                baseline_t_update = baseline["T_update_per_iter_mean"]
                impl_t_update = data["T_update_per_iter_mean"]
                
                if impl_t_update > baseline_t_update * 1.5:
                    print(f"       → Проблема в update_centroids: {impl_t_update*1000:.3f} мс vs {baseline_t_update*1000:.3f} мс")
                    if "bincount" in impl_name:
                        print(f"       → Возможная причина: цикл по D={D} создаёт накладные расходы на маленьких датасетах")
        
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    analyze_gpu_results("kmeans_timing_results.json")

