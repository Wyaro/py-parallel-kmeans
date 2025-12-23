"""
Тесты таймеров для измерения производительности.
"""

import time
import pytest
from kmeans.metrics.timers import Timer


class TestTimer:
    """Тесты контекстного менеджера Timer."""

    def test_timer_basic(self):
        """Базовый тест работы таймера."""
        with Timer() as t:
            time.sleep(0.1)

        # Проверяем, что время измерено
        assert t.elapsed > 0
        assert t.elapsed >= 0.1
        assert t.start > 0
        assert t.end > t.start

    def test_timer_elapsed_property(self):
        """Тест свойства elapsed."""
        with Timer() as t:
            time.sleep(0.05)

        # elapsed должен быть равен разности end - start
        assert abs(t.elapsed - (t.end - t.start)) < 1e-6

    def test_timer_multiple_uses(self):
        """Тест использования таймера несколько раз."""
        timer = Timer()

        with timer:
            time.sleep(0.05)

        elapsed1 = timer.elapsed

        # Второе использование должно перезаписать значения
        with timer:
            time.sleep(0.05)

        elapsed2 = timer.elapsed

        # Оба измерения должны быть положительными
        assert elapsed1 > 0
        assert elapsed2 > 0

    def test_timer_fast_operation(self):
        """Тест таймера на быстрой операции."""
        with Timer() as t:
            _ = sum(range(1000))

        # Даже быстрая операция должна занять некоторое время
        assert t.elapsed >= 0

    def test_timer_nested(self):
        """Тест вложенных таймеров."""
        with Timer() as outer:
            time.sleep(0.05)
            with Timer() as inner:
                time.sleep(0.02)

        # Внешний таймер должен измерить больше времени
        assert outer.elapsed > inner.elapsed
        assert outer.elapsed >= 0.07

