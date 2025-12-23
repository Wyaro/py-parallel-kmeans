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
        choices=[e.value for e in ExperimentId]
        + ["all"],
        default="all",
        help="Какой эксперимент запустить "
        "(all, exp1_baseline_single, exp2_scaling_n, "
        "exp3_scaling_d, exp4_scaling_k, exp5_strong_scaling, exp6_gpu_profile). "
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
        # масштабирование по N: внутри запускаются и single, и multi-process
        suite = make_suite_single()
        results = suite.run_exp2_scaling_n()
    elif args.experiment == ExperimentId.SCALING_D.value:
        # масштабирование по D на максимальном числе ядер
        suite = make_suite_mp(max_procs)
        results = suite.run_exp3_scaling_d()
    elif args.experiment == ExperimentId.SCALING_K.value:
        # масштабирование по K на максимальном числе ядер
        suite = make_suite_mp(max_procs)
        results = suite.run_exp4_scaling_k()
    elif args.experiment == ExperimentId.STRONG_SCALING.value:
        # strong scaling по числу процессов для Python multiprocessing
        suite = make_suite_single()  # модель задаётся внутри run_exp5_strong_scaling
        results = suite.run_exp5_strong_scaling()
    elif args.experiment == ExperimentId.GPU_PROFILE.value:
        suite = make_suite_single()  # dataset_cls/runner_cls те же
        results = suite.run_exp6_gpu_profile()
    else:
        raise ValueError(f"Experiment {args.experiment} is not implemented in Python runner.")

    # Пишем результаты потоково в NDJSON, чтобы не ждать окончания всех экспериментов.
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")

    logger.info(f"Finished {len(results)} experiments for '{args.experiment}'")
    logger.info(f"Timing results saved to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
