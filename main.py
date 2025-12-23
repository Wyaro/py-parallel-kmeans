"""
Основной скрипт для подготовки данных и анализа результатов.

Выполняет полный цикл:
1. Генерация синтетических датасетов
2. Визуализация датасетов
3. Анализ результатов экспериментов
"""

from pathlib import Path

from analyze_timings import compute_stats_from_results
from generate_datasets import DatasetGenerator
from vizualize_datasets import visualize_all_datasets


def main() -> None:
    """Основная функция подготовки данных и анализа результатов."""
    # Генерация датасетов
    generator = DatasetGenerator(base_seed=42)
    generator.generate_all()

    # Визуализация датасетов
    visualize_all_datasets()

    # Анализ результатов экспериментов
    root = Path(__file__).resolve().parent
    json_path = root / "timing_summary.json"
    if json_path.exists():
        compute_stats_from_results(json_path)
    else:
        print(f"Файл результатов не найден: {json_path}")
        print("Запустите эксперименты через kmeans/main.py")


if __name__ == "__main__":
    main()
