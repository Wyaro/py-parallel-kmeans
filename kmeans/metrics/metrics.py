"""
Метрики производительности для анализа результатов экспериментов.

Модуль предоставляет функции для вычисления ускорения, эффективности
и пропускной способности параллельных реализаций.
"""

from __future__ import annotations


def speedup(t_serial: float, t_parallel: float) -> float:
    """
    Вычисляет ускорение параллельной реализации относительно последовательной.

    Args:
        t_serial: Время выполнения последовательной реализации
        t_parallel: Время выполнения параллельной реализации

    Returns:
        Значение ускорения (speedup = t_serial / t_parallel)

    Raises:
        ZeroDivisionError: Если t_parallel равно нулю
    """
    if t_parallel == 0:
        raise ZeroDivisionError("Parallel time cannot be zero")
    return t_serial / t_parallel


def efficiency(speedup: float, p: int) -> float:
    """
    Вычисляет параллельную эффективность.

    Эффективность показывает, насколько хорошо используется параллелизм.
    Идеальное значение = 1.0 (линейное ускорение).

    Args:
        speedup: Значение ускорения
        p: Количество потоков/процессов

    Returns:
        Значение эффективности (efficiency = speedup / p)

    Raises:
        ZeroDivisionError: Если p равно нулю
    """
    if p == 0:
        raise ZeroDivisionError("Number of processes cannot be zero")
    return speedup / p


def throughput(
    N: int, K: int, D: int, n_iters: int, total_time: float
) -> float:
    """
    Вычисляет пропускную способность алгоритма.

    Пропускная способность = (N × K × D × n_iters) / total_time
    Показывает количество операций в секунду.

    Args:
        N: Количество точек данных
        K: Количество кластеров
        D: Размерность пространства
        n_iters: Количество итераций алгоритма
        total_time: Общее время выполнения (секунды)

    Returns:
        Пропускная способность (операций в секунду)

    Raises:
        ZeroDivisionError: Если total_time равно нулю
    """
    if total_time == 0:
        raise ZeroDivisionError("Total time cannot be zero")
    return (N * K * D * n_iters) / total_time
