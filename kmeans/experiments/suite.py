import logging
from typing import Any, Callable, Iterable, List, Type

from kmeans.experiments.config import (
    ExperimentId,
    EXPERIMENTS,
    REPEATS_SCALING_N,
    REPEATS_SCALING_D,
    REPEATS_SCALING_K,
)
from kmeans.utils.logging import format_dataset_prefix
from kmeans.core.gpu_numpy import (
    gpu_available,
    KMeansGPUCuPy,
    KMeansGPUCuPyBincount,
    KMeansGPUCuPyFast,
    KMeansGPUCuPyRaw,
)


class ExperimentSuite:
    """
    Высокоуровневый интерфейс для запуска серии экспериментов KMeans
    по различным сценариям (Exp1–Exp4 и т.д.).
    """

    def __init__(
        self,
        registry: Any,
        model_factory: Callable[..., Any],
        dataset_cls: Type[Any],
        runner_cls: Type[Any],
        logger: logging.Logger | None = None,
        max_seconds: float | None = None,
        result_sink: Callable[[dict], None] | None = None,
        gpu_only: bool = False,
    ) -> None:
        self.registry = registry
        self.model_factory = model_factory
        self.dataset_cls = dataset_cls
        self.runner_cls = runner_cls
        self.logger = logger or logging.getLogger("kmeans")
        self._gpu_ok = gpu_available()
        self.max_seconds = max_seconds
        self._sink = result_sink
        self.gpu_only = gpu_only

    def _emit(self, result: dict) -> None:
        if self._sink:
            self._sink(result)

    def _gpu_variants(self) -> list[tuple[str, Callable[..., Any]]]:
        if not self._gpu_ok:
            return []
        return [
            (
                "python_gpu_cupy",
                lambda *, n_clusters, logger=None: KMeansGPUCuPy(
                    n_clusters=n_clusters, logger=logger
                ),
            ),
            (
                "python_gpu_cupy_bincount",
                lambda *, n_clusters, logger=None: KMeansGPUCuPyBincount(
                    n_clusters=n_clusters, logger=logger
                ),
            ),
            (
                "python_gpu_cupy_fast",
                lambda *, n_clusters, logger=None: KMeansGPUCuPyFast(
                    n_clusters=n_clusters, logger=logger
                ),
            ),
            (
                "python_gpu_cupy_raw",
                lambda *, n_clusters, logger=None: KMeansGPUCuPyRaw(
                    n_clusters=n_clusters, logger=logger
                ),
            ),
        ]

    # --- Вспомогательное ---

    def _select_datasets(
        self,
        *,
        purpose: str | None = None,
        N: int | None = None,
        D: int | None = None,
        K: int | None = None,
    ) -> Iterable[dict]:
        """Фильтрация датасетов по полям summary.

        Особый случай: purpose == "base" трактуется как отсутствие поля
        purpose или явное значение "base" в summary.
        """
        for info in self.registry.get_all():
            if purpose is not None:
                entry_purpose = info.get("purpose")
                if purpose == "base":
                    if entry_purpose not in (None, "base"):
                        continue
                else:
                    if entry_purpose != purpose:
                        continue
            if N is not None and info["N"] != N:
                continue
            if D is not None and info["D"] != D:
                continue
            if K is not None and info["K"] != K:
                continue
            yield info


    def run_all(self) -> List[dict]:
        """
        Запускает все эксперименты по очереди, сохраняя оригинальные ID экспериментов
        в результатах и логах.
        """
        results: List[dict] = []

        runners = [
            ("exp1_baseline_single", self.run_exp1_baseline_single),
            ("exp2_scaling_n", self.run_exp2_scaling_n),
            ("exp3_scaling_d", self.run_exp3_scaling_d),
            ("exp4_scaling_k", self.run_exp4_scaling_k),
            ("exp5_gpu_profile", self.run_exp5_gpu_profile),
        ]

        for name, fn in runners:
            self.logger.info(f"[all] Starting {name}")
            part = fn()
            results.extend(part)
            self.logger.info(f"[all] Finished {name}: {len(part)} records")

        return results

    # --- Эксперимент 1: baseline single-thread ---

    def run_exp1_baseline_single(self) -> List[dict]:
        """
        Базовый эксперимент: только однопоточная CPU-реализация (python_cpu_numpy).
        Используется как эталон для оценки ускорения.
        """
        from kmeans.core.cpu_numpy import KMeansCPUNumpy

        cfg = EXPERIMENTS[ExperimentId.BASELINE_SINGLE]
        flt = cfg.params["filter"]

        infos = list(
            self._select_datasets(
                purpose=flt.get("purpose"),
                N=flt.get("N"),
                D=flt.get("D"),
                K=flt.get("K"),
            )
        )

        results: List[dict] = []
        for idx, info in enumerate(infos, start=1):
            prefix = format_dataset_prefix(info)
            self.logger.info(
                f"[baseline {idx}/{len(infos)}] Dataset {prefix} "
                f"filepath={info.get('filepath')}"
            )
            dataset = self.dataset_cls(self.registry.datasets_root, info)

            if not self.gpu_only:
                self.logger.info(
                    f"[baseline {idx}/{len(infos)}] Running python_cpu_numpy"
                )
                runner = self.runner_cls(
                    dataset,
                    lambda *, n_clusters, logger=None: KMeansCPUNumpy(
                        n_clusters=n_clusters, n_iters=100, logger=logger
                    ),
                    self.logger,
                )
                timing = runner.run(
                    repeats=cfg.params["repeats"],
                    warmup=cfg.params["warmup"],
                    max_seconds=self.max_seconds,
                )
                res = {
                    "experiment": cfg.id.value,
                    "implementation": "python_cpu_numpy",
                    "dataset": info,
                    "timing": timing,
                }
                results.append(res)
                self._emit(res)

        return results

    # --- Эксперимент 2: масштабирование по N ---

    def run_exp2_scaling_n(self) -> List[dict]:
        """
        Масштабирование по N: single + multiprocessing (max cores) + GPU (если доступен).
        """
        from multiprocessing import cpu_count
        from kmeans.core.cpu_numpy import KMeansCPUNumpy
        from kmeans.core.cpu_multiprocessing import (
            KMeansCPUMultiprocessing,
            MultiprocessingConfig,
        )

        cfg = EXPERIMENTS[ExperimentId.SCALING_N]
        infos = sorted(
            self._select_datasets(purpose="scaling_by_N"),
            key=lambda d: d["N"],
        )

        max_procs = cpu_count()
        results: List[dict] = []

        for idx, info in enumerate(infos, start=1):
            N = info["N"]
            repeats = REPEATS_SCALING_N.get(N, 10)
            prefix = format_dataset_prefix(info)
            self.logger.info(
                f"[scaling_N {idx}/{len(infos)}] Dataset {prefix} "
                f"repeats={repeats} filepath={info.get('filepath')}"
            )
            dataset = self.dataset_cls(self.registry.datasets_root, info)

            impls: List[tuple[str, Callable[..., Any]]] = []
            if not self.gpu_only:
                impls.extend(
                    [
                        (
                            "python_cpu_numpy",
                            lambda *, n_clusters, logger=None: KMeansCPUNumpy(
                                n_clusters=n_clusters, n_iters=100, logger=logger
                            ),
                        ),
                        (
                            f"python_cpu_mp_{max_procs}",
                            lambda *, n_clusters, logger=None: KMeansCPUMultiprocessing(
                                n_clusters=n_clusters,
                                n_iters=100,
                                mp=MultiprocessingConfig(n_processes=max_procs),
                                logger=logger,
                            ),
                        ),
                    ]
                )
            impls.extend(self._gpu_variants())

            for impl_name, factory in impls:
                self.logger.info(
                    f"[scaling_N {idx}/{len(infos)}] Running {impl_name}"
                )
                runner = self.runner_cls(dataset, factory, self.logger)
                timing = runner.run(repeats=repeats, warmup=3, max_seconds=self.max_seconds)
                res = {
                    "experiment": cfg.id.value,
                    "implementation": impl_name,
                    "threads": max_procs if impl_name.startswith("python_cpu_mp_") else None,
                    "dataset": info,
                    "timing": timing,
                }
                results.append(res)
                self._emit(res)

        return results

    # --- Эксперимент 3: масштабирование по D ---

    def run_exp3_scaling_d(self) -> List[dict]:
        cfg = EXPERIMENTS[ExperimentId.SCALING_D]
        infos = sorted(
            self._select_datasets(purpose="scaling_by_D"),
            key=lambda d: d["D"],
        )

        from multiprocessing import cpu_count
        from kmeans.core.cpu_numpy import KMeansCPUNumpy
        from kmeans.core.cpu_multiprocessing import (
            KMeansCPUMultiprocessing,
            MultiprocessingConfig,
        )

        max_procs = cpu_count()
        results: List[dict] = []
        for idx, info in enumerate(infos, start=1):
            D = info["D"]
            repeats = REPEATS_SCALING_D.get(D, 10)
            prefix = format_dataset_prefix(info)
            self.logger.info(
                f"[scaling_D {idx}/{len(infos)}] Dataset {prefix} "
                f"repeats={repeats} filepath={info.get('filepath')}"
            )
            dataset = self.dataset_cls(self.registry.datasets_root, info)

            impls: List[tuple[str, Callable[..., Any]]] = []
            if not self.gpu_only:
                impls.extend(
                    [
                        (
                            "python_cpu_numpy",
                            lambda *, n_clusters, logger=None: KMeansCPUNumpy(
                                n_clusters=n_clusters, n_iters=100, logger=logger
                            ),
                        ),
                        (
                            f"python_cpu_mp_{max_procs}",
                            lambda *, n_clusters, logger=None: KMeansCPUMultiprocessing(
                                n_clusters=n_clusters,
                                n_iters=100,
                                mp=MultiprocessingConfig(n_processes=max_procs),
                                logger=logger,
                            ),
                        ),
                    ]
                )
            impls.extend(self._gpu_variants())

            for impl_name, factory in impls:
                self.logger.info(
                    f"[scaling_D {idx}/{len(infos)}] Running {impl_name}"
                )
                runner = self.runner_cls(dataset, factory, self.logger)
                timing = runner.run(repeats=repeats, warmup=3, max_seconds=self.max_seconds)
                res = {
                    "experiment": cfg.id.value,
                    "implementation": impl_name,
                    "threads": max_procs if impl_name.startswith("python_cpu_mp_") else None,
                    "dataset": info,
                    "timing": timing,
                }
                results.append(res)
                self._emit(res)

        return results

    # --- Эксперимент 4: масштабирование по K ---

    def run_exp4_scaling_k(self) -> List[dict]:
        cfg = EXPERIMENTS[ExperimentId.SCALING_K]
        infos = sorted(
            self._select_datasets(purpose="scaling_by_K"),
            key=lambda d: d["K"],
        )

        from multiprocessing import cpu_count
        from kmeans.core.cpu_numpy import KMeansCPUNumpy
        from kmeans.core.cpu_multiprocessing import (
            KMeansCPUMultiprocessing,
            MultiprocessingConfig,
        )

        max_procs = cpu_count()
        results: List[dict] = []
        for idx, info in enumerate(infos, start=1):
            K = info["K"]
            repeats = REPEATS_SCALING_K.get(K, 10)
            prefix = format_dataset_prefix(info)
            self.logger.info(
                f"[scaling_K {idx}/{len(infos)}] Dataset {prefix} "
                f"repeats={repeats} filepath={info.get('filepath')}"
            )
            dataset = self.dataset_cls(self.registry.datasets_root, info)

            impls: List[tuple[str, Callable[..., Any]]] = []
            if not self.gpu_only:
                impls.extend(
                    [
                        (
                            "python_cpu_numpy",
                            lambda *, n_clusters, logger=None: KMeansCPUNumpy(
                                n_clusters=n_clusters, n_iters=100, logger=logger
                            ),
                        ),
                        (
                            f"python_cpu_mp_{max_procs}",
                            lambda *, n_clusters, logger=None: KMeansCPUMultiprocessing(
                                n_clusters=n_clusters,
                                n_iters=100,
                                mp=MultiprocessingConfig(n_processes=max_procs),
                                logger=logger,
                            ),
                        ),
                    ]
                )
            impls.extend(self._gpu_variants())

            for impl_name, factory in impls:
                self.logger.info(
                    f"[scaling_K {idx}/{len(infos)}] Running {impl_name}"
                )
                runner = self.runner_cls(dataset, factory, self.logger)
                timing = runner.run(repeats=repeats, warmup=3, max_seconds=self.max_seconds)
                res = {
                    "experiment": cfg.id.value,
                    "implementation": impl_name,
                    "threads": max_procs if impl_name.startswith("python_cpu_mp_") else None,
                    "dataset": info,
                    "timing": timing,
                }
                results.append(res)
                self._emit(res)

        return results

    # --- Эксперимент 5: GPU профилирование ---

    def run_exp5_gpu_profile(self) -> List[dict]:
        """
        Профилируем GPU-реализации на фиксированном крупном датасете.
        """
        cfg = EXPERIMENTS[ExperimentId.GPU_PROFILE]
        if not self._gpu_ok:
            self.logger.warning("GPU недоступен: пропуск exp5_gpu_profile")
            return []

        infos = list(self._select_datasets(N=cfg.params["N"], D=cfg.params["D"], K=cfg.params["K"]))
        if not infos:
            raise ValueError("Не найден датасет для GPU профилирования")

        info = infos[0]
        repeats = cfg.params["repeats"]
        prefix = format_dataset_prefix(info)
        self.logger.info(f"[gpu_profile] Dataset {prefix} repeats={repeats}")

        dataset = self.dataset_cls(self.registry.datasets_root, info)
        results: List[dict] = []

        for impl_name, factory in self._gpu_variants():
            self.logger.info(f"[gpu_profile] Running {impl_name}")
            runner = self.runner_cls(dataset, factory, self.logger)
            timing = runner.run(repeats=repeats, warmup=3, max_seconds=self.max_seconds)
            res = {
                "experiment": cfg.id.value,
                "implementation": impl_name,
                "dataset": info,
                "timing": timing,
            }
            results.append(res)
            self._emit(res)

        return results
