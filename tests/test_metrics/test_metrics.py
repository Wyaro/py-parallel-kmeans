"""
Тесты метрик производительности.
"""

import pytest
from kmeans.metrics.metrics import speedup, efficiency, throughput


class TestSpeedup:
    """Тесты вычисления ускорения."""

    def test_speedup_basic(self):
        """Базовый тест вычисления ускорения."""
        # Последовательное время: 10 секунд
        # Параллельное время: 5 секунд
        # Ускорение: 10/5 = 2.0
        assert speedup(10.0, 5.0) == 2.0
        assert speedup(20.0, 10.0) == 2.0

    def test_speedup_linear(self):
        """Тест линейного ускорения."""
        # Если время уменьшается в 4 раза, ускорение = 4
        assert speedup(40.0, 10.0) == 4.0
        assert speedup(100.0, 25.0) == 4.0

    def test_speedup_sublinear(self):
        """Тест сублинейного ускорения."""
        # Если время уменьшается меньше, чем в 2 раза, ускорение < 2
        result = speedup(10.0, 6.0)
        assert result == pytest.approx(10.0 / 6.0, rel=1e-10)

    def test_speedup_zero_parallel_time(self):
        """Тест обработки нулевого параллельного времени."""
        with pytest.raises(ZeroDivisionError):
            speedup(10.0, 0.0)

    def test_speedup_negative_times(self):
        """Тест с отрицательными временами (не должно быть, но проверяем)."""
        # Отрицательные времена не имеют смысла, но функция должна работать
        result = speedup(-10.0, -5.0)
        assert result == 2.0  # Математически корректно


class TestEfficiency:
    """Тесты вычисления эффективности."""

    def test_efficiency_basic(self):
        """Базовый тест вычисления эффективности."""
        # Ускорение: 2.0, количество потоков: 4
        # Эффективность: 2.0 / 4 = 0.5
        assert efficiency(2.0, 4) == 0.5
        assert efficiency(4.0, 4) == 1.0  # Идеальная эффективность

    def test_efficiency_perfect(self):
        """Тест идеальной эффективности."""
        # Линейное ускорение: ускорение = количество потоков
        assert efficiency(4.0, 4) == 1.0
        assert efficiency(8.0, 8) == 1.0

    def test_efficiency_suboptimal(self):
        """Тест неоптимальной эффективности."""
        # Ускорение меньше количества потоков
        assert efficiency(2.0, 8) == 0.25
        assert efficiency(3.0, 6) == 0.5

    def test_efficiency_zero_processes(self):
        """Тест обработки нулевого количества процессов."""
        with pytest.raises(ZeroDivisionError):
            efficiency(2.0, 0)

    def test_efficiency_greater_than_one(self):
        """Тест эффективности больше 1 (суперлинейное ускорение)."""
        # Суперлинейное ускорение возможно (например, из-за кэша)
        result = efficiency(10.0, 8)
        assert result == 1.25


class TestThroughput:
    """Тесты вычисления пропускной способности."""

    def test_throughput_basic(self):
        """Базовый тест вычисления пропускной способности."""
        # 1000 точек, 2 кластера, 2D, 10 итераций за 2 секунды
        # Пропускная способность = (1000 * 2 * 2 * 10) / 2 = 20000 операций/сек
        result = throughput(1000, 2, 2, 10, 2.0)
        assert result == 20000.0

    def test_throughput_simple(self):
        """Простой тест пропускной способности."""
        # 100 точек, 1 кластер, 1D, 1 итерация за 1 секунду
        # = (100 * 1 * 1 * 1) / 1 = 100 операций/сек
        assert throughput(100, 1, 1, 1, 1.0) == 100.0

    def test_throughput_large_dataset(self):
        """Тест пропускной способности на большом датасете."""
        # 1M точек, 10 кластеров, 50D, 100 итераций за 10 секунд
        N, K, D, n_iters, time = 1_000_000, 10, 50, 100, 10.0
        result = throughput(N, K, D, n_iters, time)
        expected = (N * K * D * n_iters) / time
        assert result == expected

    def test_throughput_zero_time(self):
        """Тест обработки нулевого времени."""
        with pytest.raises(ZeroDivisionError):
            throughput(1000, 2, 2, 10, 0.0)

    def test_throughput_consistency(self):
        """Тест согласованности формулы."""
        # Проверяем, что формула работает корректно для разных значений
        test_cases = [
            (100, 2, 3, 5, 1.0),
            (500, 4, 10, 20, 5.0),
            (1000, 8, 50, 100, 10.0),
        ]

        for N, K, D, n_iters, time in test_cases:
            result = throughput(N, K, D, n_iters, time)
            expected = (N * K * D * n_iters) / time
            assert result == pytest.approx(expected, rel=1e-10)

