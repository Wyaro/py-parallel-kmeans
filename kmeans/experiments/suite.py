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
        results: List[dict] = []
        datasets_info: Iterable[dict] = list(self.registry.get_all())

        for idx, info in enumerate(datasets_info, start=1):
            prefix = format_dataset_prefix(info)
            self.logger.info(
                f"[{idx}/{len(datasets_info)}] Dataset {prefix} "
                f"filepath={info.get('filepath')}"
            )
            dataset = self.dataset_cls(self.registry.datasets_root, info)
            
            # CPU реализация - пропускаем если gpu_only
            if not self.gpu_only:
                runner = self.runner_cls(dataset, self.model_factory, self.logger)
                timing = runner.run(max_seconds=self.max_seconds)

                res = {
                    "experiment": "all",
                    "implementation": "python_cpu_numpy",
                    "dataset": info,
                    "timing": timing,
                }
                results.append(res)
                self._emit(res)

            # GPU реализации
            for impl_name, factory in self._gpu_variants():
                self.logger.info(
                    f"[{idx}/{len(datasets_info)}] Running {impl_name}"
                )
                runner_gpu = self.runner_cls(dataset, factory, self.logger)
                timing_gpu = runner_gpu.run(max_seconds=self.max_seconds)
                res_gpu = {
                    "experiment": "all",
                    "implementation": impl_name,
                    "dataset": info,
                    "timing": timing_gpu,
                }
                results.append(res_gpu)
                self._emit(res_gpu)

        return results

    # --- Эксперимент 1: baseline single-thread ---

    def run_exp1_baseline_single(self) -> List[dict]:
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
            
            # CPU реализация - пропускаем если gpu_only
            if not self.gpu_only:
                runner = self.runner_cls(dataset, self.model_factory, self.logger)
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

            # GPU реализации
            for impl_name, factory in self._gpu_variants():
                self.logger.info(
                    f"[baseline {idx}/{len(infos)}] Running {impl_name}"
                )
                runner_gpu = self.runner_cls(dataset, factory, self.logger)
                timing_gpu = runner_gpu.run(
                    repeats=cfg.params["repeats"],
                    warmup=cfg.params["warmup"],
                    max_seconds=self.max_seconds,
                )
                res_gpu = {
                    "experiment": cfg.id.value,
                    "implementation": impl_name,
                    "dataset": info,
                    "timing": timing_gpu,
                }
                results.append(res_gpu)
                self._emit(res_gpu)

        return results

    # --- Эксперимент 2: масштабирование по N ---

    def run_exp2_scaling_n(self) -> List[dict]:
        """
        Масштабирование по N для двух реализаций:
        - однопоточная (KMeansCPUNumpy);
        - многопроцессная (KMeansCPUMultiprocessing) на максимуме ядер.
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

            # 1) Однопоточная реализация (serial baseline) - пропускаем если gpu_only
            if not self.gpu_only:
                self.logger.info(
                    f"[scaling_N {idx}/{len(infos)}] "
                    f"Running single-thread implementation (python_cpu_numpy)"
                )

                def single_model_factory(*, n_clusters: int, logger=None) -> Any:
                    return KMeansCPUNumpy(
                        n_clusters=n_clusters,
                        n_iters=100,
                        logger=logger,
                    )

                runner_single = self.runner_cls(dataset, single_model_factory, self.logger)
                timing_single = runner_single.run(repeats=repeats, warmup=3, max_seconds=self.max_seconds)

                res_single = {
                    "experiment": cfg.id.value,
                    "implementation": "python_cpu_numpy",
                    "dataset": info,
                    "timing": timing_single,
                }
                results.append(res_single)
                self._emit(res_single)

                # 2) Многопроцессная реализация на максимуме доступных ядер
                self.logger.info(
                    f"[scaling_N {idx}/{len(infos)}] "
                    f"Running multi-process implementation "
                    f"(python_cpu_mp_{max_procs}, threads={max_procs})"
                )

                def mp_model_factory(*, n_clusters: int, logger=None) -> Any:
                    return KMeansCPUMultiprocessing(
                        n_clusters=n_clusters,
                        n_iters=100,
                        mp=MultiprocessingConfig(n_processes=max_procs),
                        logger=logger,
                    )

                runner_mp = self.runner_cls(dataset, mp_model_factory, self.logger)
                timing_mp = runner_mp.run(repeats=repeats, warmup=3, max_seconds=self.max_seconds)

                res_mp = {
                    "experiment": cfg.id.value,
                    "implementation": f"python_cpu_mp_{max_procs}",
                    "threads": max_procs,
                    "dataset": info,
                    "timing": timing_mp,
                }
                results.append(res_mp)
                self._emit(res_mp)

            # 3) GPU реализации (если доступны)
            for impl_name, factory in self._gpu_variants():
                self.logger.info(
                    f"[scaling_N {idx}/{len(infos)}] Running {impl_name}"
                )
                runner_gpu = self.runner_cls(dataset, factory, self.logger)
                timing_gpu = runner_gpu.run(repeats=repeats, warmup=3, max_seconds=self.max_seconds)
                res_gpu = {
                    "experiment": cfg.id.value,
                    "implementation": impl_name,
                    "dataset": info,
                    "timing": timing_gpu,
                }
                results.append(res_gpu)
                self._emit(res_gpu)

        return results

    # --- Эксперимент 3: масштабирование по D ---

    def run_exp3_scaling_d(self) -> List[dict]:
        cfg = EXPERIMENTS[ExperimentId.SCALING_D]
        infos = sorted(
            self._select_datasets(purpose="scaling_by_D"),
            key=lambda d: d["D"],
        )

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
            
            # CPU реализация - пропускаем если gpu_only
            if not self.gpu_only:
                runner = self.runner_cls(dataset, self.model_factory, self.logger)
                timing = runner.run(repeats=repeats, warmup=3, max_seconds=self.max_seconds)

                res = {
                    "experiment": cfg.id.value,
                    "implementation": "python_cpu_numpy",
                    "dataset": info,
                    "timing": timing,
                }
                results.append(res)
                self._emit(res)

            # GPU реализации
            for impl_name, factory in self._gpu_variants():
                self.logger.info(
                    f"[scaling_D {idx}/{len(infos)}] Running {impl_name}"
                )
                runner_gpu = self.runner_cls(dataset, factory, self.logger)
                timing_gpu = runner_gpu.run(repeats=repeats, warmup=3, max_seconds=self.max_seconds)
                res_gpu = {
                    "experiment": cfg.id.value,
                    "implementation": impl_name,
                    "dataset": info,
                    "timing": timing_gpu,
                }
                results.append(res_gpu)
                self._emit(res_gpu)

        return results

    # --- Эксперимент 4: масштабирование по K ---

    def run_exp4_scaling_k(self) -> List[dict]:
        cfg = EXPERIMENTS[ExperimentId.SCALING_K]
        infos = sorted(
            self._select_datasets(purpose="scaling_by_K"),
            key=lambda d: d["K"],
        )

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
            
            # CPU реализация - пропускаем если gpu_only
            if not self.gpu_only:
                runner = self.runner_cls(dataset, self.model_factory, self.logger)
                timing = runner.run(repeats=repeats, warmup=3, max_seconds=self.max_seconds)

                res = {
                    "experiment": cfg.id.value,
                    "implementation": "python_cpu_numpy",
                    "dataset": info,
                    "timing": timing,
                }
                results.append(res)
                self._emit(res)

            # GPU реализации
            for impl_name, factory in self._gpu_variants():
                self.logger.info(
                    f"[scaling_K {idx}/{len(infos)}] Running {impl_name}"
                )
                runner_gpu = self.runner_cls(dataset, factory, self.logger)
                timing_gpu = runner_gpu.run(repeats=repeats, warmup=3, max_seconds=self.max_seconds)
                res_gpu = {
                    "experiment": cfg.id.value,
                    "implementation": impl_name,
                    "dataset": info,
                    "timing": timing_gpu,
                }
                results.append(res_gpu)
                self._emit(res_gpu)

        return results

    # --- Эксперимент 5: strong scaling для multiprocessing-реализации ---

    def run_exp5_strong_scaling(self) -> List[dict]:
        """
        Strong scaling по числу процессов для одного большого датасета N=1e6,D=50,K=8.

        Для Python-реализации используется KMeansCPUMultiprocessing с разным
        числом процессов; остальные реализации (C++/C#) предполагаются внешними.
        """
        from multiprocessing import cpu_count
        from kmeans.core.cpu_multiprocessing import (
            KMeansCPUMultiprocessing,
            MultiprocessingConfig,
        )

        cfg = EXPERIMENTS[ExperimentId.STRONG_SCALING]

        N = cfg.params["N"]
        D = cfg.params["D"]
        K = cfg.params["K"]
        threads_list = cfg.params["threads"]
        repeats = cfg.params["repeats"]

        # выбираем датасет с такими N,D,K (purpose не фиксируем)
        infos = list(self._select_datasets(N=N, D=D, K=K))
        if not infos:
            raise ValueError(f"No dataset found for strong scaling N={N}, D={D}, K={K}")

        info = infos[0]
        prefix = format_dataset_prefix(info)
        self.logger.info(f"[strong_scaling] Dataset {prefix} filepath={info.get('filepath')}")

        dataset = self.dataset_cls(self.registry.datasets_root, info)
        results: List[dict] = []

        max_procs = cpu_count()

        # CPU multiprocessing реализации - пропускаем если gpu_only
        if not self.gpu_only:
            for n_procs in threads_list:
                n_procs_eff = max(1, min(int(n_procs), max_procs))

                self.logger.info(
                    f"[strong_scaling] Python multiprocessing with n_procs={n_procs_eff}"
                )

                def mp_model_factory(*, n_clusters: int, logger=None) -> Any:
                    return KMeansCPUMultiprocessing(
                        n_clusters=n_clusters,
                        n_iters=100,
                        mp=MultiprocessingConfig(n_processes=n_procs_eff),
                        logger=logger,
                    )

                runner = self.runner_cls(dataset, mp_model_factory, self.logger)
                timing = runner.run(repeats=repeats, warmup=3, max_seconds=self.max_seconds)

                res = {
                    "experiment": cfg.id.value,
                    "implementation": f"python_cpu_mp_{n_procs_eff}",
                    "threads": n_procs_eff,
                    "dataset": info,
                    "timing": timing,
                }
                results.append(res)
                self._emit(res)

        # GPU реализации для strong scaling
        for impl_name, factory in self._gpu_variants():
            self.logger.info(f"[strong_scaling] Running {impl_name}")
            runner_gpu = self.runner_cls(dataset, factory, self.logger)
            timing_gpu = runner_gpu.run(repeats=repeats, warmup=3, max_seconds=self.max_seconds)
            res_gpu = {
                "experiment": cfg.id.value,
                "implementation": impl_name,
                "dataset": info,
                "timing": timing_gpu,
            }
            results.append(res_gpu)
            self._emit(res_gpu)

        return results

    # --- Эксперимент 6: GPU профилирование ---

    def run_exp6_gpu_profile(self) -> List[dict]:
        """
        Профилируем GPU-реализации на фиксированном крупном датасете.
        """
        cfg = EXPERIMENTS[ExperimentId.GPU_PROFILE]
        if not self._gpu_ok:
            self.logger.warning("GPU недоступен: пропуск exp6_gpu_profile")
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
