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
    KMeansGPUCuPyV1,
    KMeansGPUCuPyV2,
    KMeansGPUCuPyV3,
    KMeansGPUCuPyV4,
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

    def _calculate_metrics(
        self,
        result: dict,
        baseline_result: dict | None = None,
    ) -> dict:
        """
        Вычисляет дополнительные метрики производительности:
        - Ускорение (Speedup): S = T_опр / T_мпр
        - Параллельная эффективность: E = S / p
        - Пропускная способность: ПС = (N × K × D × N_итер) / T_общ
        - Время передачи данных (GPU): T_пер = T_H2D + T_D2H
        - Доля времени на передачу данных (GPU): R_пер = T_пер / T_общ × 100%

        :param result: результат эксперимента для многопоточной реализации
        :param baseline_result: результат эксперимента для однопоточной реализации (baseline)
        :return: словарь с дополнительными метриками
        """
        timing = result.get("timing", {})
        dataset = result.get("dataset", {})
        impl_name = result.get("implementation", "")
        
        N = dataset.get("N", 0)
        K = dataset.get("K", 0)
        D = dataset.get("D", 0)
        
        T_общ = timing.get("T_fit_avg", 0.0)
        
        metrics = {}
        
        # 1. Ускорение (Speedup): S = T_опр / T_мпр
        if baseline_result is not None:
            baseline_timing = baseline_result.get("timing", {})
            T_опр = baseline_timing.get("T_fit_avg", 0.0)
            if T_общ > 0 and T_опр > 0:
                speedup = T_опр / T_общ
                metrics["speedup"] = float(speedup)
            else:
                metrics["speedup"] = None
        else:
            metrics["speedup"] = None
        
        # 2. Параллельная эффективность: E = S / p
        # p - число потоков (CPU) или количество блоков (GPU)
        p = None
        
        # Для CPU multiprocessing
        if "threads" in result:
            p = result["threads"]
        # Для GPU - вычисляем количество блоков
        elif impl_name.startswith("python_gpu_cupy"):
            # Для GPU вычисляем количество блоков на основе N
            # threads_per_block = 256 (стандартное значение)
            threads_per_block = 256
            if N > 0:
                blocks = (N + threads_per_block - 1) // threads_per_block
                p = blocks
                metrics["gpu_blocks"] = blocks
                metrics["gpu_threads_per_block"] = threads_per_block
        
        if metrics.get("speedup") is not None and p is not None and p > 0:
            efficiency = metrics["speedup"] / p
            metrics["efficiency"] = float(efficiency)
        else:
            metrics["efficiency"] = None
        
        if p is not None:
            metrics["parallelism_factor"] = p
        
        # 3. Пропускная способность: ПС = (N × K × D × N_итер) / T_общ
        # Используем среднее количество итераций из runs
        runs = timing.get("runs", [])
        if runs:
            n_iters_actual_list = [r.get("n_iters_actual", 0) for r in runs]
            n_iters_avg = sum(n_iters_actual_list) / len(n_iters_actual_list) if n_iters_actual_list else 0
        else:
            n_iters_avg = 0
        
        if T_общ > 0 and N > 0 and K > 0 and D > 0 and n_iters_avg > 0:
            throughput = (N * K * D * n_iters_avg) / T_общ
            metrics["throughput"] = float(throughput)
        else:
            metrics["throughput"] = None
        
        # 4. Время передачи данных (GPU): T_пер = T_H2D + T_D2H
        # Уже вычисляется в runner.py как T_transfer
        T_transfer = timing.get("T_transfer_avg", None)
        if T_transfer is not None:
            metrics["T_transfer"] = float(T_transfer)
        else:
            metrics["T_transfer"] = None
        
        # 5. Доля времени на передачу данных (GPU): R_пер = T_пер / T_общ × 100%
        # Уже вычисляется в runner.py как T_transfer_ratio
        T_transfer_ratio = timing.get("T_transfer_ratio_avg", None)
        if T_transfer_ratio is not None:
            metrics["T_transfer_ratio"] = float(T_transfer_ratio)
        else:
            metrics["T_transfer_ratio"] = None
        
        return metrics

    def _gpu_variants(self, dataset_info: dict | None = None) -> list[tuple[str, Callable[..., Any]]]:
        """
        Возвращает список GPU-вариантов для экспериментов.
        
        Все версии (V1-V4) теперь поддерживают большие датасеты благодаря
        оптимизированным алгоритмам без материализации больших промежуточных массивов.
        
        :param dataset_info: опциональная информация о датасете (N, K, D)
        """
        if not self._gpu_ok:
            return []
        
        variants = []
        
        # V1 теперь использует оптимизированную формулу без материализации diff
        # и может работать с любыми размерами датасетов
        variants.append((
            "python_gpu_cupy_v1",
            lambda *, n_clusters, logger=None: KMeansGPUCuPyV1(
                n_clusters=n_clusters, logger=logger
            ),
        ))
        
        variants.extend([
            (
                "python_gpu_cupy_v2",
                lambda *, n_clusters, logger=None: KMeansGPUCuPyV2(
                    n_clusters=n_clusters, logger=logger
                ),
            ),
            (
                "python_gpu_cupy_v3",
                lambda *, n_clusters, logger=None: KMeansGPUCuPyV3(
                    n_clusters=n_clusters, logger=logger
                ),
            ),
            (
                "python_gpu_cupy_v4",
                lambda *, n_clusters, logger=None: KMeansGPUCuPyV4(
                    n_clusters=n_clusters, logger=logger
                ),
            ),
        ])
        
        return variants

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
            for impl_name, factory in self._gpu_variants(dataset_info=info):
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

    # --- Эксперимент 1: baseline single-thread (только однопоточная Python) ---

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

            # Только однопоточная Python-реализация (baseline) — без multiprocessing и GPU.
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
                # Вычисляем метрики производительности относительно однопоточной реализации
                metrics = self._calculate_metrics(res_mp, baseline_result=res_single)
                res_mp["metrics"] = metrics
                results.append(res_mp)
                self._emit(res_mp)

            # 3) GPU реализации (если доступны)
            for impl_name, factory in self._gpu_variants(dataset_info=info):
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
                # Вычисляем метрики производительности относительно однопоточной реализации
                # Если gpu_only, baseline_result будет None, и метрики не будут вычислены
                baseline_for_gpu = res_single if not self.gpu_only else None
                metrics = self._calculate_metrics(res_gpu, baseline_result=baseline_for_gpu)
                res_gpu["metrics"] = metrics
                results.append(res_gpu)
                self._emit(res_gpu)

        return results

    # --- Эксперимент 3: масштабирование по D ---

    def run_exp3_scaling_d(self) -> List[dict]:
        """
        Масштабирование по D для однопоточной, multiprocessing и GPU реализаций.
        """
        from multiprocessing import cpu_count
        from kmeans.core.cpu_numpy import KMeansCPUNumpy
        from kmeans.core.cpu_multiprocessing import (
            KMeansCPUMultiprocessing,
            MultiprocessingConfig,
        )

        cfg = EXPERIMENTS[ExperimentId.SCALING_D]
        infos = sorted(
            self._select_datasets(purpose="scaling_by_D"),
            key=lambda d: d["D"],
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

            res_single = None
            # 1) Однопоточная реализация (baseline) - пропускаем если gpu_only
            if not self.gpu_only:
                self.logger.info(
                    f"[scaling_D {idx}/{len(infos)}] "
                    f"Running single-thread implementation (python_cpu_numpy)"
                )

                def single_model_factory(*, n_clusters: int, logger=None) -> Any:
                    return KMeansCPUNumpy(
                        n_clusters=n_clusters,
                        n_iters=100,
                        logger=logger,
                    )

                runner_single = self.runner_cls(dataset, single_model_factory, self.logger)
                timing_single = runner_single.run(
                    repeats=repeats, warmup=3, max_seconds=self.max_seconds
                )

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
                    f"[scaling_D {idx}/{len(infos)}] "
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
                timing_mp = runner_mp.run(
                    repeats=repeats, warmup=3, max_seconds=self.max_seconds
                )

                res_mp = {
                    "experiment": cfg.id.value,
                    "implementation": f"python_cpu_mp_{max_procs}",
                    "threads": max_procs,
                    "dataset": info,
                    "timing": timing_mp,
                }
                metrics_mp = self._calculate_metrics(res_mp, baseline_result=res_single)
                res_mp["metrics"] = metrics_mp
                results.append(res_mp)
                self._emit(res_mp)

            # 3) GPU реализации
            for impl_name, factory in self._gpu_variants(dataset_info=info):
                self.logger.info(
                    f"[scaling_D {idx}/{len(infos)}] Running {impl_name}"
                )
                runner_gpu = self.runner_cls(dataset, factory, self.logger)
                timing_gpu = runner_gpu.run(
                    repeats=repeats, warmup=3, max_seconds=self.max_seconds
                )
                res_gpu = {
                    "experiment": cfg.id.value,
                    "implementation": impl_name,
                    "dataset": info,
                    "timing": timing_gpu,
                }
                baseline_for_gpu = res_single if not self.gpu_only else None
                metrics_gpu = self._calculate_metrics(
                    res_gpu, baseline_result=baseline_for_gpu
                )
                res_gpu["metrics"] = metrics_gpu
                results.append(res_gpu)
                self._emit(res_gpu)

        return results

    # --- Эксперимент 4: масштабирование по K ---

    def run_exp4_scaling_k(self) -> List[dict]:
        """
        Масштабирование по K для однопоточной, multiprocessing и GPU реализаций.
        """
        from multiprocessing import cpu_count
        from kmeans.core.cpu_numpy import KMeansCPUNumpy
        from kmeans.core.cpu_multiprocessing import (
            KMeansCPUMultiprocessing,
            MultiprocessingConfig,
        )

        cfg = EXPERIMENTS[ExperimentId.SCALING_K]
        infos = sorted(
            self._select_datasets(purpose="scaling_by_K"),
            key=lambda d: d["K"],
        )

        max_procs = cpu_count()
        results: List[dict] = []

        for idx, info in enumerate(infos, start=1):
            K_val = info["K"]
            repeats = REPEATS_SCALING_K.get(K_val, 10)
            prefix = format_dataset_prefix(info)
            self.logger.info(
                f"[scaling_K {idx}/{len(infos)}] Dataset {prefix} "
                f"repeats={repeats} filepath={info.get('filepath')}"
            )
            dataset = self.dataset_cls(self.registry.datasets_root, info)

            res_single = None
            # 1) Однопоточная реализация (baseline) - пропускаем если gpu_only
            if not self.gpu_only:
                self.logger.info(
                    f"[scaling_K {idx}/{len(infos)}] "
                    f"Running single-thread implementation (python_cpu_numpy)"
                )

                def single_model_factory(*, n_clusters: int, logger=None) -> Any:
                    return KMeansCPUNumpy(
                        n_clusters=n_clusters,
                        n_iters=100,
                        logger=logger,
                    )

                runner_single = self.runner_cls(dataset, single_model_factory, self.logger)
                timing_single = runner_single.run(
                    repeats=repeats, warmup=3, max_seconds=self.max_seconds
                )

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
                    f"[scaling_K {idx}/{len(infos)}] "
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
                timing_mp = runner_mp.run(
                    repeats=repeats, warmup=3, max_seconds=self.max_seconds
                )

                res_mp = {
                    "experiment": cfg.id.value,
                    "implementation": f"python_cpu_mp_{max_procs}",
                    "threads": max_procs,
                    "dataset": info,
                    "timing": timing_mp,
                }
                metrics_mp = self._calculate_metrics(res_mp, baseline_result=res_single)
                res_mp["metrics"] = metrics_mp
                results.append(res_mp)
                self._emit(res_mp)

            # 3) GPU реализации
            for impl_name, factory in self._gpu_variants(dataset_info=info):
                self.logger.info(
                    f"[scaling_K {idx}/{len(infos)}] Running {impl_name}"
                )
                runner_gpu = self.runner_cls(dataset, factory, self.logger)
                timing_gpu = runner_gpu.run(
                    repeats=repeats, warmup=3, max_seconds=self.max_seconds
                )
                res_gpu = {
                    "experiment": cfg.id.value,
                    "implementation": impl_name,
                    "dataset": info,
                    "timing": timing_gpu,
                }
                baseline_for_gpu = res_single if not self.gpu_only else None
                metrics_gpu = self._calculate_metrics(
                    res_gpu, baseline_result=baseline_for_gpu
                )
                res_gpu["metrics"] = metrics_gpu
                results.append(res_gpu)
                self._emit(res_gpu)

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

        for impl_name, factory in self._gpu_variants(dataset_info=info):
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
