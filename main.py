from pathlib import Path

from analyze_timings import compute_stats_from_results
from generate_datasets import DatasetGenerator
from vizualize_datasets import visualize_all_datasets


def main():
    """Основная функция"""
    #generator = DatasetGenerator(base_seed=42)
    #generator.generate_all()
    #visualize_all_datasets()

    root = Path(__file__).resolve().parent
    json_path = root / "timing_summary.json"
    compute_stats_from_results(json_path)

if __name__ == "__main__":
    main()