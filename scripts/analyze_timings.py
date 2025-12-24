"""
Анализ результатов экспериментов по производительности K-means.

Модуль читает результаты таймингов из JSON / NDJSON файла и выводит
статистику по времени выполнения различных этапов алгоритма K-means.

Поддерживаемые форматы файла результатов:
- JSON-массив объектов (старый формат);
- NDJSON (каждая строка — отдельный JSON-объект, новый формат).
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import argparse


def _determine_experiment(exp_id: str, dataset_info: dict[str, Any]) -> str:
    """
    Определяет конкретный эксперимент по ID и параметрам датасета.
    
    Если exp_id == "all", пытается определить конкретный эксперимент
    по параметрам датасета (N, D, K, purpose).
    
    Args:
        exp_id: ID эксперимента из результатов (может быть "all")
        dataset_info: Информация о датасете (N, D, K, purpose)
    
    Returns:
        Конкретный ID эксперимента (exp1_baseline_single, exp2_scaling_n, и т.д.)
    """
    if exp_id != "all":
        return exp_id
    
    N = dataset_info.get("N")
    D = dataset_info.get("D")
    K = dataset_info.get("K")
    purpose = dataset_info.get("purpose")
    
    # Эксперимент 1: baseline single
    if N == 100_000 and D == 50 and K == 8 and purpose == "base":
        return "exp1_baseline_single"
    
    # Эксперимент 5: GPU profile (проверяем раньше exp2, т.к. N=1_000_000 может быть в обоих)
    # exp5 обычно имеет N=1_000_000, D=50, K=8, но без purpose="scaling_by_N"
    if N == 1_000_000 and D == 50 and K == 8 and purpose != "scaling_by_N":
        return "exp5_gpu_profile"
    
    # Эксперимент 2: scaling by N
    scaling_n_values = {1_000, 100_000, 1_000_000, 5_000_000}
    if purpose == "scaling_by_N" or (N in scaling_n_values and D == 50 and K == 8):
        return "exp2_scaling_n"
    
    # Эксперимент 3: scaling by D
    scaling_d_values = {2, 10, 50, 100, 200}
    if purpose == "scaling_by_D" or (D in scaling_d_values and N == 100_000 and K == 8):
        return "exp3_scaling_d"
    
    # Эксперимент 4: scaling by K
    scaling_k_values = {4, 8, 16, 32}
    if purpose == "scaling_by_K" or (K in scaling_k_values and N == 100_000 and D == 50):
        return "exp4_scaling_k"
    
    # Если не удалось определить, возвращаем "all"
    return exp_id


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
        self.json_path: Path = Path(json_path)
        self.n_iters: int = n_iters

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
        # поэтому делим на реальное количество итераций (n_iters_actual) для получения времени одной итерации
        t_assign_ms = []
        t_update_ms = []
        t_iter_ms = []
        t_total_ms = []
        
        for r in runs:
            # Используем реальное количество итераций из прогона, если доступно
            # Иначе используем значение по умолчанию из конструктора
            n_iters_actual = r.get("n_iters_actual")
            if n_iters_actual is None or n_iters_actual <= 0:
                n_iters_actual = self.n_iters
            
            n_iters_actual = float(n_iters_actual)
            
            # Время одной итерации (в миллисекундах)
            if n_iters_actual > 0:
                t_assign_ms.append(float(r["T_assign_total"]) * 1000.0 / n_iters_actual)
                t_update_ms.append(float(r["T_update_total"]) * 1000.0 / n_iters_actual)
                t_iter_ms.append(float(r["T_iter_total"]) * 1000.0 / n_iters_actual)
            else:
                # Если итераций не было, используем 0
                t_assign_ms.append(0.0)
                t_update_ms.append(0.0)
                t_iter_ms.append(0.0)
            
            # T_fit — это общее время одного запуска алгоритма (в миллисекундах)
            t_total_ms.append(float(r["T_fit"]) * 1000.0)

        if not t_assign_ms:
            # Если нет данных, возвращаем нули
            return {
                "assign_mean": 0.0,
                "assign_med": 0.0,
                "update_mean": 0.0,
                "update_med": 0.0,
                "iter_mean": 0.0,
                "iter_med": 0.0,
                "total_mean": 0.0,
                "total_med": 0.0,
            }

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

    def analyze(self, *, output_path: Path | str | None = None) -> None:
        """
        Анализирует результаты и выводит статистику в консоль.

        Для каждого эксперимента выводит компактную таблицу со статистикой.
        Если указан output_path, дополнительно пишет сводку в файл (UTF-8).
        """
        # Группируем результаты по эксперименту и датасету
        grouped: dict[tuple[str, int, int, int], list[dict[str, Any]]] = {}
        
        for entry in self._iter_results():
            exp = entry.get("experiment", "unknown")
            dataset_info: dict[str, Any] = entry.get("dataset", {}) or {}
            
            exp_determined = _determine_experiment(exp, dataset_info)
            N = dataset_info.get("N", 0)
            D = dataset_info.get("D", 0)
            K = dataset_info.get("K", 0)
            
            key = (exp_determined, N, D, K)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(entry)
        
        lines: list[str] = []
        
        # Сортируем по эксперименту и параметрам датасета
        for (exp_determined, N, D, K), entries in sorted(grouped.items()):
            # Заголовок для группы
            print(f"\n{'='*80}")
            print(f"{exp_determined} | N={N:,} D={D} K={K}")
            print(f"{'='*80}")
            
            # Заголовок таблицы
            header = (
                f"{'Реализация':<30} "
                f"{'Итер':<6} "
                f"{'T_назн (мс)':<15} "
                f"{'T_обн (мс)':<15} "
                f"{'T_итер (мс)':<15} "
                f"{'T_общ (мс)':<15}"
            )
            print(header)
            print("-" * 80)
            
            lines.append(f"{exp_determined} | N={N:,} D={D} K={K}")
            lines.append("=" * 80)
            lines.append(header)
            lines.append("-" * 80)
            
            # Данные для каждой реализации
            for entry in sorted(entries, key=lambda e: e.get("implementation", "")):
                impl = entry.get("implementation", "unknown")
                timing: dict[str, Any] = entry.get("timing", {}) or {}
                runs = timing.get("runs") or []
                
                if not runs:
                    continue
                
                stats = self._compute_statistics(runs)
                
                # Вычисляем среднее количество итераций
                n_iters_list = [
                    float(r.get("n_iters_actual", self.n_iters))
                    for r in runs
                    if r.get("n_iters_actual") is not None
                ]
                avg_n_iters = mean(n_iters_list) if n_iters_list else self.n_iters
                
                # Форматируем значения: показываем среднее (медиана в скобках)
                assign_str = f"{stats['assign_mean']:.3f} ({stats['assign_med']:.3f})"
                update_str = f"{stats['update_mean']:.3f} ({stats['update_med']:.3f})"
                iter_str = f"{stats['iter_mean']:.3f} ({stats['iter_med']:.3f})"
                total_str = f"{stats['total_mean']:.3f} ({stats['total_med']:.3f})"
                
                row = (
                    f"{impl:<30} "
                    f"{avg_n_iters:>5.1f} "
                    f"{assign_str:>15} "
                    f"{update_str:>15} "
                    f"{iter_str:>15} "
                    f"{total_str:>15}"
                )
                print(row)
                lines.append(row)
            
            print()
            lines.append("")

        if output_path is not None:
            out = Path(output_path)
            out.write_text("\n".join(lines), encoding="utf-8")


def compute_stats_from_results(
    json_path: str | Path, n_iters: int = 100, *, output_path: str | Path | None = None
) -> None:
    """
    Удобная функция для быстрого анализа результатов.

    Args:
        json_path: Путь к файлу с результатами
        n_iters: Количество итераций алгоритма
        output_path: Путь для сохранения текстовой сводки (опционально)
    """
    analyzer = TimingResultsAnalyzer(json_path, n_iters)
    analyzer.analyze(output_path=output_path)


def _cli() -> None:
    """CLI-обёртка для анализа таймингов."""
    parser = argparse.ArgumentParser(
        description=(
            "Анализ результатов таймингов K-means из JSON/NDJSON "
            "и вывод агрегированной статистики."
        )
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Путь к файлу с результатами таймингов (JSON/NDJSON)",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=100,
        help=(
            "Количество итераций алгоритма по умолчанию "
            "(если не указано в самих результатах, по умолчанию: 100)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Путь к текстовому файлу для сохранения сводки (опционально)",
    )

    args = parser.parse_args()
    compute_stats_from_results(
        json_path=args.json_path, n_iters=args.n_iters, output_path=args.output
    )


if __name__ == "__main__":
    _cli()
