"""
Высокоточные таймеры для измерения производительности.

Модуль предоставляет контекстный менеджер Timer для измерения времени
выполнения участков кода с использованием time.perf_counter().
"""
from __future__ import annotations
import time
from typing import Any


class Timer:
    """
    Контекстный менеджер для измерения времени выполнения кода.

    Использует time.perf_counter() для высокоточных измерений времени,
    не зависящих от системных часов.

    Пример использования:
        with Timer() as t:
            # код для измерения
            pass
        elapsed_time = t.elapsed
    """

    def __init__(self) -> None:
        """Инициализация таймера."""
        self.start: float = 0.0
        self.end: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> Timer:
        """
        Вход в контекстный менеджер.

        Returns:
            Экземпляр Timer
        """
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        """
        Выход из контекстного менеджера.

        Вычисляет прошедшее время и сохраняет в self.elapsed.
        """
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
