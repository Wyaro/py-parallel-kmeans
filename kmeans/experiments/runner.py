import logging
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from kmeans.metrics.timers import Timer
from kmeans.utils.logging import format_dataset_prefix


class _PrefixedLogger:
    """Обёртка над логгером, добавляющая префикс к каждому сообщению."""

    def __init__(self, base_logger: logging.Logger | None, prefix: str) -> None:
        self._base = base_logger
        self._prefix = prefix

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self._base:
            self._base.info(f"{self._prefix} {msg}", *args, **kwargs)


class ExperimentRunner:
    """
    Запускает серию прогонов KMeans на одном датасете.

    Ожидается, что снаружи будет передан:
    - dataset: экземпляр Dataset
    - model_factory: callable, создающий модель KMeans по параметрам датасета
    """

    def __init__(
        self,
        dataset: Any,
        model_factory: Callable[..., Any],
        logger: logging.Logger | None = None,
    ) -> None:
        self.dataset = dataset
        self.model_factory = model_factory
        self.logger = logger

        meta: Dict[str, Any] = self.dataset.dataset_info
        self._dataset_prefix = format_dataset_prefix(meta)

    def _create_model(self) -> Any:
        """Создаёт новую модель под размерность текущего датасета."""
        K = self.dataset.dataset_info["K"]
        # Передаём в модель логгер с префиксом датасета,
        # чтобы сообщения вида "Iteration X/Y" были понятны.
        logger = _PrefixedLogger(self.logger, self._dataset_prefix)
        return self.model_factory(n_clusters=K, logger=logger)

    def run(
        self,
        repeats: int = 100,
        warmup: int = 3,
        max_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Запускает несколько прогонов KMeans с таймингом.

        :param repeats: количество измеряемых прогонов
        :param warmup: количество «разогревочных» запусков
        :param max_seconds: лимит стендового времени (warmup + измеряемые прогоны).
            Если прогнозируемое время превысит лимит, цикл прерывается, а
            результаты дополняются оценкой оставшегося времени.
        :return: словарь с агрегированной статистикой времени
        """
        X = self.dataset.X
        centroids = self.dataset.initial_centroids

        if self.logger:
            self.logger.info(f"{self._dataset_prefix} Warmup x{warmup}")

        warmup_start = time.perf_counter()
        # тёплый прогон (не учитываем в финальной статистике, но учитываем во времени стены)
        for _ in range(warmup):
            model = self._create_model()
            model.fit(X, centroids)
        warmup_elapsed = time.perf_counter() - warmup_start

        times: List[float] = []
        assign_totals: List[float] = []
        update_totals: List[float] = []
        iter_totals: List[float] = []
        runs: List[Dict[str, float]] = []
        throughput_ops: List[float] = []
        transfer_totals: List[float] = []
        transfer_ratios: List[float] = []

        estimated = False
        estimated_total_seconds: float | None = None

        for run_idx in range(1, repeats + 1):
            if self.logger:
                self.logger.info(f"{self._dataset_prefix} Run {run_idx}/{repeats}")

            model = self._create_model()
            with Timer() as t_fit:
                model.fit(X, centroids)
            t_fit_val = float(t_fit.elapsed)
            times.append(t_fit_val)

            # низкоуровневые тайминги шага назначения и обновления
            t_assign = float(model.t_assign_total)
            t_update = float(model.t_update_total)
            t_iter = float(model.t_iter_total)

            assign_totals.append(t_assign)
            update_totals.append(t_update)
            iter_totals.append(t_iter)

            runs.append(
                {
                    "run_idx": float(run_idx),
                    "T_fit": t_fit_val,
                    "T_assign_total": t_assign,
                    "T_update_total": t_update,
                    "T_iter_total": t_iter,
                    "n_iters_actual": int(model.n_iters_actual),
                    # Пропускная способность (операции/с) — N*K*D*n_iters_actual / T_fit
                    "throughput_ops": float(
                        self.dataset.dataset_info["N"]
                        * self.dataset.dataset_info["K"]
                        * self.dataset.dataset_info["D"]
                        * int(model.n_iters_actual)
                        / t_fit_val
                    ),
                    # Время передачи для GPU (если модель его измерила)
                    "T_transfer": float(
                        getattr(model, "t_h2d", 0.0) + getattr(model, "t_d2h", 0.0)
                    ),
                    "T_transfer_ratio": float(
                        ((getattr(model, "t_h2d", 0.0) + getattr(model, "t_d2h", 0.0)) / t_fit_val)
                        * 100.0
                        if t_fit_val > 0.0
                        else 0.0
                    ),
                }
            )
            throughput_ops.append(runs[-1]["throughput_ops"])
            transfer_totals.append(runs[-1]["T_transfer"])
            transfer_ratios.append(runs[-1]["T_transfer_ratio"])

            if max_seconds is not None and times:
                avg_time = float(sum(times) / len(times))
                remaining = (repeats - run_idx) * avg_time
                spent = warmup_elapsed + sum(times)
                if spent + remaining > max_seconds:
                    estimated = True
                    estimated_total_seconds = spent + remaining
                    if self.logger:
                        self.logger.warning(
                            f"{self._dataset_prefix} Ранний выход по лимиту времени: "
                            f"spent={spent:.2f}s, remaining_est={remaining:.2f}s, "
                            f"limit={max_seconds:.2f}s"
                        )
                    break

        stats: Dict[str, Any] = {
            # полное время одного запуска KMeans (по внешнему таймеру)
            "T_fit_avg": float(np.mean(times)),
            "T_fit_std": float(np.std(times)),
            "T_fit_min": float(np.min(times)),
            # агрегированные времена по шагам внутри fit
            # T_назначения, T_обновления, T_итерации (сумма двух)
            "T_assign_total_avg": float(np.mean(assign_totals)),
            "T_update_total_avg": float(np.mean(update_totals)),
            "T_iter_total_avg": float(np.mean(iter_totals)),
            # агрегация по дополнительным метрикам
            "throughput_ops_avg": float(np.mean(throughput_ops)),
            "throughput_ops_med": float(np.median(throughput_ops)),
            "T_transfer_avg": float(np.mean(transfer_totals)),
            "T_transfer_med": float(np.median(transfer_totals)),
            "T_transfer_ratio_avg": float(np.mean(transfer_ratios)),
            "T_transfer_ratio_med": float(np.median(transfer_ratios)),
            # подробные метрики по каждому запуску
            "runs": runs,
            # информация об оценке/лимите
            "estimated": estimated,
            "repeats_done": len(times),
            "repeats_requested": repeats,
            "estimated_total_seconds": estimated_total_seconds,
            "warmup_seconds": warmup_elapsed,
            "time_spent_seconds": warmup_elapsed + sum(times),
        }

        if self.logger:
            self.logger.info(
                f"{self._dataset_prefix} Timing: "
                f"T_fit_avg={stats['T_fit_avg']:.6f}s, "
                f"T_fit_std={stats['T_fit_std']:.6f}s, "
                f"T_fit_min={stats['T_fit_min']:.6f}s, "
                f"T_assign_total_avg={stats['T_assign_total_avg']:.6f}s, "
                f"T_update_total_avg={stats['T_update_total_avg']:.6f}s"
            )

        return stats
