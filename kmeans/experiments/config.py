from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List


class ExperimentId(str, Enum):
    BASELINE_SINGLE = "exp1_baseline_single"
    SCALING_N = "exp2_scaling_n"
    SCALING_D = "exp3_scaling_d"
    SCALING_K = "exp4_scaling_k"
    GPU_PROFILE = "exp5_gpu_profile"


@dataclass
class ExperimentConfig:
    id: ExperimentId
    description: str
    implementations: List[str]
    params: Dict[str, Any]


# Таблицы соответствия N/D/K → числа повторов из постановки задачи

REPEATS_SCALING_N: Dict[int, int] = {
    1_000: 50,
    100_000: 20,
    1_000_000: 10,
    5_000_000: 5,
}

REPEATS_SCALING_D: Dict[int, int] = {
    2: 20,
    10: 20,
    50: 10,
    100: 10,
    200: 10,
}

REPEATS_SCALING_K: Dict[int, int] = {
    4: 20,
    8: 20,
    16: 10,
    32: 10,
}


EXPERIMENTS: Dict[ExperimentId, ExperimentConfig] = {
    ExperimentId.BASELINE_SINGLE: ExperimentConfig(
        id=ExperimentId.BASELINE_SINGLE,
        description="Эксперимент 1: реализация однопоточных реализаций N=100000,D=50,K=8",
        implementations=["python_cpu_numpy", "cpp_single", "csharp_single"],
        params={
            # берём ровно один базовый датасет (purpose="base")
            "filter": {"N": 100_000, "D": 50, "K": 8, "purpose": "base"},
            "repeats": 30,
            "warmup": 3,
        },
    ),
    ExperimentId.SCALING_N: ExperimentConfig(
        id=ExperimentId.SCALING_N,
        description="Эксперимент 2: масштабирование по N",
        implementations=["python_cpu_numpy", "python_cpu_mt", "cpp_openmp", "csharp_tpl", "gpu"],
        params={
            "purpose": "scaling_by_N",
            "repeats_by_N": REPEATS_SCALING_N,
        },
    ),
    ExperimentId.SCALING_D: ExperimentConfig(
        id=ExperimentId.SCALING_D,
        description="Эксперимент 3: масштабирование по D",
        implementations=["python_cpu_numpy", "python_cpu_mt", "cpp_openmp", "csharp_tpl", "gpu"],
        params={
            "purpose": "scaling_by_D",
            "repeats_by_D": REPEATS_SCALING_D,
        },
    ),
    ExperimentId.SCALING_K: ExperimentConfig(
        id=ExperimentId.SCALING_K,
        description="Эксперимент 4: масштабирование по K",
        implementations=["python_cpu_numpy", "python_cpu_mt", "cpp_openmp", "csharp_tpl", "gpu"],
        params={
            "purpose": "scaling_by_K",
            "repeats_by_K": REPEATS_SCALING_K,
        },
    ),
    ExperimentId.GPU_PROFILE: ExperimentConfig(
        id=ExperimentId.GPU_PROFILE,
        description="Эксперимент 5: GPU профилирование (Nsight Compute)",
        implementations=["gpu"],
        params={
            "N": 1_000_000,
            "D": 50,
            "K": 8,
            "repeats": 10,
        },
    ),
}


