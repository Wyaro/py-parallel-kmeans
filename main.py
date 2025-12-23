"""
Основной скрипт для управления полным циклом работы с K-means экспериментами.

Поддерживает три режима работы:
1. Генерация датасетов и визуализация
2. Проведение экспериментов
3. Сбор аналитики результатов
"""

import argparse
import json
import sys
from multiprocessing import cpu_count
from pathlib import Path

from scripts.analyze_timings import compute_stats_from_results
from scripts.generate_datasets import DatasetGenerator
from scripts.vizualize_datasets import visualize_all_datasets


def run_datasets_generation(visualize: bool = True) -> None:
    """
    Генерирует синтетические датасеты и выполняет их визуализацию.
    
    Args:
        visualize: Выполнять ли визуализацию датасетов
    """
    print("=" * 80)
    print("ГЕНЕРАЦИЯ ДАТАСЕТОВ И ВИЗУАЛИЗАЦИЯ")
    print("=" * 80)
    print()
    
    print("Шаг 1: Генерация синтетических датасетов...")
    generator = DatasetGenerator(base_seed=42)
    generator.generate_all()
    print("   Датасеты сгенерированы")
    print()
    
    if visualize:
        print("Шаг 2: Визуализация датасетов...")
        visualize_all_datasets()
        print("   Визуализация завершена")
        print()
    
    print("=" * 80)
    print("ГЕНЕРАЦИЯ ДАТАСЕТОВ ЗАВЕРШЕНА")
    print("=" * 80)


def run_experiments(
    experiment: str = "all",
    max_seconds: float = 1800.0,
    gpu_only: bool = False,
    auto_analyze: bool = False,
) -> None:
    """
    Запускает эксперименты по производительности K-means.
    
    Args:
        experiment: Идентификатор эксперимента (all, exp1_baseline_single, и т.д.)
        max_seconds: Лимит времени на выполнение экспериментов
        gpu_only: Запускать только GPU реализации
        auto_analyze: Автоматически запустить анализ после завершения экспериментов
    """
    print("=" * 80)
    print("ПРОВЕДЕНИЕ ЭКСПЕРИМЕНТОВ")
    print("=" * 80)
    print()
    
    # Импортируем необходимые модули для запуска экспериментов
    from kmeans.data.registry import DatasetRegistry
    from kmeans.data.dataset import Dataset
    from kmeans.core.cpu_numpy import KMeansCPUNumpy
    from kmeans.core.cpu_multiprocessing import (
        KMeansCPUMultiprocessing,
        MultiprocessingConfig,
    )
    from kmeans.experiments.config import ExperimentId
    from kmeans.experiments.runner import ExperimentRunner
    from kmeans.experiments.suite import ExperimentSuite
    from kmeans.utils.logging import setup_logger
    
    root = Path(__file__).resolve().parent
    datasets = root / "datasets"
    summary = datasets / "datasets_summary.json"
    results_json = root / "kmeans_timing_results.json"
    
    logger = setup_logger()
    
    # Проверка доступности GPU при использовании --gpu-only
    if gpu_only:
        from kmeans.core.gpu_numpy import gpu_available
        if not gpu_available():
            logger.error(
                "GPU недоступен, но указан флаг --gpu-only. "
                "Установите CuPy или уберите флаг."
            )
            return
        logger.info("Режим --gpu-only: будут запущены только GPU реализации")
    
    registry = DatasetRegistry(
        summary_path=summary,
        datasets_root=datasets,
    )
    
    # Потоковая запись результатов в NDJSON
    results_json.write_text("", encoding="utf-8")
    
    def sink_writer(rec: dict) -> None:
        with open(results_json, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
    
    def make_suite_single() -> ExperimentSuite:
        return ExperimentSuite(
            registry=registry,
            model_factory=lambda **kw: KMeansCPUNumpy(n_iters=100, **kw),
            dataset_cls=Dataset,
            runner_cls=ExperimentRunner,
            logger=logger,
            max_seconds=max_seconds,
            result_sink=sink_writer,
            gpu_only=gpu_only,
        )
    
    def make_suite_mp(n_procs: int) -> ExperimentSuite:
        return ExperimentSuite(
            registry=registry,
            model_factory=lambda **kw: KMeansCPUMultiprocessing(
                n_iters=100,
                mp=MultiprocessingConfig(n_processes=n_procs),
                **kw,
            ),
            dataset_cls=Dataset,
            runner_cls=ExperimentRunner,
            logger=logger,
            max_seconds=max_seconds,
            result_sink=sink_writer,
            gpu_only=gpu_only,
        )
    
    max_procs = cpu_count()
    
    # Запуск соответствующего эксперимента
    if experiment == "all":
        suite = make_suite_single()
        results = suite.run_all()
    elif experiment == ExperimentId.BASELINE_SINGLE.value:
        suite = make_suite_single()
        results = suite.run_exp1_baseline_single()
    elif experiment == ExperimentId.SCALING_N.value:
        suite = make_suite_single()
        results = suite.run_exp2_scaling_n()
    elif experiment == ExperimentId.SCALING_D.value:
        suite = make_suite_mp(max_procs)
        results = suite.run_exp3_scaling_d()
    elif experiment == ExperimentId.SCALING_K.value:
        suite = make_suite_mp(max_procs)
        results = suite.run_exp4_scaling_k()
    elif experiment == ExperimentId.GPU_PROFILE.value:
        suite = make_suite_single()
        results = suite.run_exp5_gpu_profile()
    else:
        raise ValueError(f"Experiment {experiment} is not implemented.")
    
    # Сохранение результатов
    with open(results_json, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
    
    logger.info(f"Finished {len(results)} experiments for '{experiment}'")
    logger.info(f"Timing results saved to {results_json}")
    
    print()
    print("=" * 80)
    print("ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
    print("=" * 80)
    
    # Автоматический запуск аналитики, если запрошено
    if auto_analyze:
        print()
        run_analysis(results_json)


def run_analysis(results_json: Path | None = None, analysis_output: Path | None = None) -> None:
    """
    Запускает полный анализ результатов экспериментов.
    
    Args:
        results_json: Путь к файлу с результатами экспериментов (NDJSON).
                     Если None, используется kmeans_timing_results.json
        analysis_output: Путь для сохранения текстовой сводки (опционально)
    """
    print("=" * 80)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТОВ")
    print("=" * 80)
    print()
    
    root = Path(__file__).resolve().parent
    if results_json is None:
        results_json = root / "kmeans_timing_results.json"
    else:
        results_json = Path(results_json)
    
    if not results_json.exists():
        print(f"[ERROR] Файл результатов не найден: {results_json}")
        print("   Запустите эксперименты через: python main.py experiments")
        return
    
    # 1. Анализ таймингов
    print("Шаг 1: Анализ таймингов...")
    output_path = analysis_output or results_json.parent / "analysis_summary.txt"
    compute_stats_from_results(
        results_json,
        n_iters=100,  # Будет использовано реальное значение из данных
        output_path=output_path,
    )
    print(f"   Результаты сохранены в: {output_path}")
    print()
    
    # 2. Расчет метрик производительности (если есть analysis_summary.txt)
    summary_path = results_json.parent / "analysis_summary.txt"
    if summary_path.exists():
        print("Шаг 2: Расчет метрик производительности...")
        metrics_output = results_json.parent / "metrics_summary.txt"
        
        from scripts.calculate_metrics_from_summary import (
            calculate_metrics,
            format_metrics_output,
            parse_summary_file,
        )
        
        results = parse_summary_file(str(summary_path))
        metrics_results = calculate_metrics(results)
        output = format_metrics_output(metrics_results)
        
        with open(metrics_output, 'w', encoding='utf-8') as f:
            f.write(output)
        
        print(f"   Метрики сохранены в: {metrics_output}")
        print()
    
    print("=" * 80)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 80)


def main() -> None:
    """Основная функция управления полным циклом работы."""
    parser = argparse.ArgumentParser(
        description="Управление полным циклом работы с K-means экспериментами",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Генерация датасетов и визуализация
  python main.py datasets

  # Проведение всех экспериментов
  python main.py experiments

  # Проведение конкретного эксперимента с автоматическим анализом
  python main.py experiments --experiment exp2_scaling_n --analyze

  # Только анализ результатов
  python main.py analysis

  # Анализ с указанием файла результатов
  python main.py analysis --results-file path/to/results.json
        """,
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Режим работы')
    
    # Режим генерации датасетов
    datasets_parser = subparsers.add_parser(
        'datasets',
        help='Генерация синтетических датасетов и их визуализация'
    )
    datasets_parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Пропустить визуализацию датасетов'
    )
    
    # Режим проведения экспериментов
    experiments_parser = subparsers.add_parser(
        'experiments',
        help='Проведение экспериментов по производительности'
    )
    experiments_parser.add_argument(
        '--experiment',
        type=str,
        choices=['all', 'exp1_baseline_single', 'exp2_scaling_n', 
                 'exp3_scaling_d', 'exp4_scaling_k', 'exp5_gpu_profile'],
        default='all',
        help='Идентификатор эксперимента (по умолчанию: all)'
    )
    experiments_parser.add_argument(
        '--max-seconds',
        type=float,
        default=1800.0,
        help='Лимит времени на выполнение экспериментов в секундах (по умолчанию: 1800)'
    )
    experiments_parser.add_argument(
        '--gpu-only',
        action='store_true',
        help='Запускать только GPU реализации (пропустить CPU алгоритмы)'
    )
    experiments_parser.add_argument(
        '--analyze',
        action='store_true',
        help='Автоматически запустить анализ результатов после завершения экспериментов'
    )
    
    # Режим анализа результатов
    analysis_parser = subparsers.add_parser(
        'analysis',
        help='Анализ результатов экспериментов'
    )
    analysis_parser.add_argument(
        '--results-file',
        type=str,
        default=None,
        help='Путь к файлу с результатами (по умолчанию: kmeans_timing_results.json)'
    )
    analysis_parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Путь для сохранения текстовой сводки анализа'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'datasets':
        run_datasets_generation(visualize=not args.no_visualize)
    elif args.mode == 'experiments':
        run_experiments(
            experiment=args.experiment,
            max_seconds=args.max_seconds,
            gpu_only=args.gpu_only,
            auto_analyze=args.analyze,
        )
    elif args.mode == 'analysis':
        results_file = Path(args.results_file) if args.results_file else None
        output = Path(args.output) if args.output else None
        run_analysis(results_json=results_file, analysis_output=output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
