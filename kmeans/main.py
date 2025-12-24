    # main.py
from pathlib import Path
import argparse
import json
from multiprocessing import cpu_count

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

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
SUMMARY = DATASETS / "datasets_summary.json"
RESULTS_JSON = ROOT / "kmeans_timing_results.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        choices=[e.value for e in ExperimentId] + ["all"],
        default="all",
        help="Какой эксперимент запустить "
        "(all, exp1_baseline_single, exp2_scaling_n, "
        "exp3_scaling_d, exp4_scaling_k, exp5_gpu_profile). "
        "Используйте --gpu-only для запуска только GPU реализаций.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=1800.0,
        help="Лимит времени (в секундах) на warmup+замеры; "
        "при прогнозе превышения делаем ранний выход с оценкой.",
    )
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Запускать только GPU реализации (пропустить CPU алгоритмы).",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Автоматически запустить анализ результатов после завершения экспериментов.",
    )
    args = parser.parse_args()

    logger = setup_logger()
    
    # Проверка доступности GPU при использовании --gpu-only
    if args.gpu_only:
        from kmeans.core.gpu_numpy import gpu_available
        if not gpu_available():
            logger.error("GPU недоступен, но указан флаг --gpu-only. Установите CuPy или уберите флаг.")
            return
        logger.info("Режим --gpu-only: будут запущены только GPU реализации")
    
    registry = DatasetRegistry(
        summary_path=SUMMARY,
        datasets_root=DATASETS,
    )

    # Потоковая запись результатов в NDJSON, чтобы не ждать окончания всех запусков.
    # Перед стартом очищаем файл.
    RESULTS_JSON.write_text("", encoding="utf-8")
    def sink_writer(rec: dict) -> None:
        with open(RESULTS_JSON, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")

    def make_suite_single() -> ExperimentSuite:
        return ExperimentSuite(
            registry=registry,
            model_factory=lambda **kw: KMeansCPUNumpy(
                n_iters=100,
                **kw,
            ),
            dataset_cls=Dataset,
            runner_cls=ExperimentRunner,
            logger=logger,
            max_seconds=args.max_seconds,
            result_sink=sink_writer,
            gpu_only=args.gpu_only,
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
            max_seconds=args.max_seconds,
            result_sink=sink_writer,
            gpu_only=args.gpu_only,
        )

    max_procs = cpu_count()

    if args.experiment == "all":
        suite = make_suite_single()
        results = suite.run_all()
    elif args.experiment == ExperimentId.BASELINE_SINGLE.value:
        suite = make_suite_single()
        results = suite.run_exp1_baseline_single()
    elif args.experiment == ExperimentId.SCALING_N.value:
        # Масштабирование по N: внутри запускаются single, multiprocessing и GPU
        suite = make_suite_single()
        results = suite.run_exp2_scaling_n()
    elif args.experiment == ExperimentId.SCALING_D.value:
        # Масштабирование по D: single, multiprocessing и GPU
        suite = make_suite_single()
        results = suite.run_exp3_scaling_d()
    elif args.experiment == ExperimentId.SCALING_K.value:
        # Масштабирование по K: single, multiprocessing и GPU
        suite = make_suite_single()
        results = suite.run_exp4_scaling_k()
    elif args.experiment == ExperimentId.GPU_PROFILE.value:
        suite = make_suite_single()  # dataset_cls/runner_cls те же
        results = suite.run_exp5_gpu_profile()
    else:
        raise ValueError(f"Experiment {args.experiment} is not implemented in Python runner.")

    # Пишем результаты потоково в NDJSON, чтобы не ждать окончания всех экспериментов.
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")

    logger.info(f"Finished {len(results)} experiments for '{args.experiment}'")
    logger.info(f"Timing results saved to {RESULTS_JSON}")
    
    # Автоматический запуск аналитики, если запрошено
    if args.analyze:
        logger.info("Starting automatic analysis of results...")
        try:
            # Импортируем функции анализа напрямую из scripts
            from scripts.analyze_timings import compute_stats_from_results
            from scripts.calculate_metrics_from_summary import (
                calculate_metrics,
                format_metrics_output,
                parse_summary_file,
            )
            
            # Шаг 1: Анализ таймингов
            analysis_output = ROOT / "analysis_summary.txt"
            compute_stats_from_results(
                RESULTS_JSON,
                n_iters=100,
                output_path=analysis_output,
            )
            logger.info(f"Timing analysis saved to {analysis_output}")
            
            # Шаг 2: Расчет метрик (если есть analysis_summary.txt)
            if analysis_output.exists():
                metrics_output = ROOT / "metrics_summary.txt"
                results = parse_summary_file(str(analysis_output))
                metrics_results = calculate_metrics(results)
                output = format_metrics_output(metrics_results)
                
                with open(metrics_output, 'w', encoding='utf-8') as f:
                    f.write(output)
                
                logger.info(f"Metrics analysis saved to {metrics_output}")
            
            logger.info("Analysis completed successfully")
        except Exception as e:
            logger.error(f"Failed to run analysis: {e}")
            logger.info("You can run analysis manually with: python main.py --analysis-only")


if __name__ == "__main__":
    main()
