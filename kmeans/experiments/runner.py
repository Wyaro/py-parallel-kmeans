import logging
from typing import Any, Callable, Dict, List

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

    def run(self, repeats: int = 100, warmup: int = 3) -> Dict[str, Any]:
        """
        Запускает несколько прогонов KMeans с таймингом.

        :param repeats: количество измеряемых прогонов
        :param warmup: количество «разогревочных» запусков
        :return: словарь с агрегированной статистикой времени
        """
        X = self.dataset.X
        centroids = self.dataset.initial_centroids

        if self.logger:
            self.logger.info(f"{self._dataset_prefix} Warmup x{warmup}")

        # тёплый прогон (не учитываем во времени)
        for _ in range(warmup):
            model = self._create_model()
            model.fit(X, centroids)

        times: List[float] = []
        assign_totals: List[float] = []
        update_totals: List[float] = []
        iter_totals: List[float] = []
        runs: List[Dict[str, float]] = []

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
                }
            )

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
            # подробные метрики по каждому запуску
            "runs": runs,
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
