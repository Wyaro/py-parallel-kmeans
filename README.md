# K-Means Benchmark Suite

Комплексная система для экспериментального исследования и сравнения вычислительной эффективности различных реализаций алгоритма K-means на CPU и GPU. Проект включает генерацию синтетических датасетов, множественные реализации алгоритма кластеризации, автоматизированные эксперименты по масштабированию производительности и инструменты для анализа результатов.

## Содержание

- [Описание](#описание)
- [Основные возможности](#основные-возможности)
- [Архитектура проекта](#архитектура-проекта)
- [Установка и требования](#установка-и-требования)
- [Быстрый старт](#быстрый-старт)
- [Использование](#использование)
- [Структура проекта](#структура-проекта)
- [Эксперименты](#эксперименты)
- [Реализации алгоритма](#реализации-алгоритма)
- [Анализ результатов](#анализ-результатов)
- [Расширение проекта](#расширение-проекта)

---

## Описание

Проект предназначен для количественной оценки вычислительной эффективности различных реализаций алгоритма K-means и выявления условий, при которых применение той или иной архитектуры вычислений является наиболее целесообразным.

### Цели исследования

- Сравнительный анализ производительности однопоточных, многопоточных и GPU реализаций
- Исследование масштабируемости по параметрам N (количество точек), D (размерность), K (количество кластеров)
- Выявление узких мест производительности для каждой реализации
- Формирование практических рекомендаций по выбору архитектуры вычислений

### Особенности реализации

- Множественные реализации: CPU (NumPy, multiprocessing), GPU (CuPy/CUDA)
- Автоматизированные эксперименты: 6 типов экспериментов с различными параметрами
- Генерация датасетов: синтетические данные с контролируемыми параметрами
- Потоковая запись результатов: результаты сохраняются по мере выполнения
- Ограничение времени: автоматический ранний выход при превышении лимита
- Детальная статистика: анализ времени выполнения по этапам алгоритма

---

## Основные возможности

### Генерация датасетов

Создание синтетических датасетов с различными параметрами для экспериментов:

```python
from generate_datasets import DatasetGenerator

generator = DatasetGenerator(base_seed=42)
generator.generate_all()  # Генерирует все типы датасетов
```

### Запуск экспериментов

Автоматизированные эксперименты по производительности:

```bash
# Все эксперименты
python -m kmeans.main --experiment all

# Конкретный эксперимент
python -m kmeans.main --experiment exp2_scaling_n

# С ограничением времени (30 минут)
python -m kmeans.main --experiment exp2_scaling_n --max-seconds 1800

# Только GPU реализации (пропустить CPU алгоритмы)
python -m kmeans.main --experiment exp2_scaling_n --gpu-only
```

### Анализ результатов

Статистический анализ времени выполнения:

```python
from analyze_timings import compute_stats_from_results

compute_stats_from_results("kmeans_timing_results.json")
```

---

## Архитектура проекта

Проект построен на принципах объектно-ориентированного программирования с чётким разделением ответственности:

```
┌─────────────────────────────────────────────────────────┐
│                    Эксперименты                        │
│  (suite.py, runner.py, config.py)                      │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│              Реализации K-Means                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ CPU NumPy    │  │ CPU MultiProc│  │ GPU CuPy    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│              Управление данными                         │
│  (Dataset, DatasetRegistry, validation)                 │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│              Метрики и таймеры                         │
│  (metrics.py, timers.py)                                │
└─────────────────────────────────────────────────────────┘
```

### Ключевые компоненты

- **`kmeans/core/`**: Реализации алгоритма K-means (базовый класс, CPU, GPU)
- **`kmeans/data/`**: Управление датасетами (загрузка, реестр, валидация)
- **`kmeans/experiments/`**: Оркестрация экспериментов (конфигурация, запуск, сбор результатов)
- **`kmeans/metrics/`**: Метрики производительности (ускорение, эффективность, пропускная способность)
- **`kmeans/utils/`**: Вспомогательные утилиты (логирование, форматирование)

---

## Установка и требования

### Системные требования

- Python 3.10 или выше
- Операционная система: Windows, Linux, macOS
- Для GPU-реализаций: NVIDIA GPU с поддержкой CUDA

### Зависимости

Основные зависимости:
- NumPy >= 1.20.0
- SciKit-learn >= 1.0.0 (для генерации датасетов)
- Matplotlib >= 3.3.0 (для визуализации)

Опциональные зависимости:
- CuPy >= 12.0.0 (для GPU-реализаций, требует CUDA)
- pytest >= 7.0.0 (для тестирования)
- pytest-cov >= 4.0.0 (для анализа покрытия кода)

### Установка

#### Быстрая установка

```bash
# Клонирование репозитория
git clone <repository-url>
cd parallel

# Создание виртуального окружения
python -m venv .venv

# Активация окружения
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Для GPU-реализаций (опционально, требуется CUDA)
pip install cupy-cuda12x  # или cupy-cuda11x в зависимости от версии CUDA
```

#### Настройка в PyCharm

Подробная инструкция по настройке виртуального окружения в PyCharm и других IDE доступна в файле [SETUP.md](SETUP.md).

**Краткая инструкция**:
1. Откройте проект в PyCharm
2. При появлении уведомления "No Python interpreter configured" нажмите "Configure Python Interpreter"
3. Создайте новое виртуальное окружение в `.venv`
4. PyCharm автоматически обнаружит `requirements.txt` и предложит установить зависимости
5. Для GPU поддержки установите CuPy: `pip install cupy-cuda12x` (или `cupy-cuda11x` в зависимости от версии CUDA)

### Проверка установки

```python
# Проверка базовых зависимостей
import numpy as np
import sklearn
print(f"NumPy версия: {np.__version__}")
print(f"SciKit-learn версия: {sklearn.__version__}")

# Проверка доступности GPU
from kmeans.core.gpu_numpy import gpu_available
print(f"GPU доступен: {gpu_available()}")
```

---

## Быстрый старт

### Шаг 1: Генерация датасетов

```bash
python main.py
```

Это создаст:
- Базовые датасеты для экспериментов
- Датасеты для масштабирования по N, D, K
- Валидационные датасеты
- Визуализации всех датасетов

### Шаг 2: Запуск экспериментов

```bash
# Базовый эксперимент (однопоточная реализация)
python -m kmeans.main --experiment exp1_baseline_single

# Масштабирование по количеству точек
python -m kmeans.main --experiment exp2_scaling_n
```

### Шаг 3: Анализ результатов

```bash
python -c "from analyze_timings import compute_stats_from_results; compute_stats_from_results('kmeans_timing_results.json')"
```

---

## Использование

### Генерация датасетов

```python
from generate_datasets import DatasetGenerator, DatasetConfig

# Создание генератора
generator = DatasetGenerator(base_seed=42)

# Генерация конкретного типа датасетов
generator.generate_scaling_N()  # Масштабирование по N
generator.generate_scaling_D()  # Масштабирование по D
generator.generate_scaling_K()  # Масштабирование по K

# Или все сразу
generator.generate_all()
```

### Использование реализаций K-means

```python
import numpy as np
from kmeans.core.cpu_numpy import KMeansCPUNumpy
from kmeans.core.cpu_multiprocessing import (
    KMeansCPUMultiprocessing,
    MultiprocessingConfig
)
from kmeans.core.gpu_numpy import KMeansGPUCuPy, gpu_available

# Подготовка данных
X = np.random.randn(10000, 50)
initial_centroids = np.random.randn(8, 50)

# Однопоточная реализация
model = KMeansCPUNumpy(n_clusters=8, n_iters=100)
model.fit(X, initial_centroids)
print(f"Центроиды: {model.centroids.shape}")
print(f"Метки: {model.labels.shape}")

# Многопроцессная реализация
mp_config = MultiprocessingConfig(n_processes=4)
model_mp = KMeansCPUMultiprocessing(n_clusters=8, mp=mp_config)
model_mp.fit(X, initial_centroids)

# GPU реализация (если доступна)
if gpu_available():
    model_gpu = KMeansGPUCuPy(n_clusters=8)
    model_gpu.fit(X, initial_centroids)
```

### Загрузка датасетов

```python
from kmeans.data.registry import DatasetRegistry
from kmeans.data.dataset import Dataset

# Инициализация реестра
registry = DatasetRegistry(
    summary_path="datasets/datasets_summary.json",
    datasets_root="datasets"
)

# Поиск датасета
dataset_info = registry.find(N=100000, D=50, K=8, purpose="base")

# Загрузка датасета
dataset = Dataset("datasets", dataset_info["metadata"])
print(f"Данные: {dataset.X.shape}")
print(f"Центроиды: {dataset.initial_centroids.shape}")
```

### Запуск экспериментов программно

```python
from kmeans.experiments.suite import ExperimentSuite
from kmeans.experiments.runner import ExperimentRunner
from kmeans.data.registry import DatasetRegistry
from kmeans.data.dataset import Dataset
from kmeans.core.cpu_numpy import KMeansCPUNumpy

registry = DatasetRegistry("datasets/datasets_summary.json", "datasets")

suite = ExperimentSuite(
    registry=registry,
    model_factory=lambda **kw: KMeansCPUNumpy(n_iters=100, **kw),
    dataset_cls=Dataset,
    runner_cls=ExperimentRunner,
)

# Запуск эксперимента
results = suite.run_exp1_baseline_single()
```

---

## Структура проекта

```
parallel/
├── main.py                      # Основной скрипт подготовки данных
├── generate_datasets.py          # Генератор синтетических датасетов
├── vizualize_datasets.py        # Визуализация датасетов
├── analyze_timings.py           # Анализ результатов экспериментов
│
├── datasets/                     # Сгенерированные датасеты
│   ├── base/                    # Базовые датасеты
│   ├── scaling_N/               # Масштабирование по N
│   ├── scaling_D/               # Масштабирование по D
│   ├── scaling_K/               # Масштабирование по K
│   ├── validation/              # Валидационные датасеты
│   └── datasets_summary.json    # Сводная информация
│
├── kmeans/                      # Основной пакет
│   ├── main.py                  # CLI для запуска экспериментов
│   │
│   ├── core/                    # Реализации алгоритма
│   │   ├── base.py              # Базовый абстрактный класс
│   │   ├── cpu_numpy.py         # Однопоточная CPU реализация
│   │   ├── cpu_multiprocessing.py  # Многопроцессная CPU реализация
│   │   └── gpu_numpy.py         # GPU реализация (CuPy/CUDA)
│   │
│   ├── data/                    # Управление данными
│   │   ├── dataset.py           # Класс Dataset
│   │   ├── registry.py          # Реестр датасетов
│   │   └── validation.py        # Валидация датасетов
│   │
│   ├── experiments/             # Эксперименты
│   │   ├── config.py            # Конфигурация экспериментов
│   │   ├── runner.py            # Запуск экспериментов
│   │   └── suite.py             # Оркестрация экспериментов
│   │
│   ├── metrics/                 # Метрики производительности
│   │   ├── metrics.py           # Ускорение, эффективность, пропускная способность
│   │   └── timers.py            # Высокоточные таймеры
│   │
│   └── utils/                   # Вспомогательные утилиты
│       └── logging.py           # Настройка логирования
│
├── kmeans_timing_results.json  # Результаты экспериментов (NDJSON)
├── visualizations/              # Визуализации датасетов
│
├── tests/                       # Тесты проекта
│   ├── conftest.py             # Фикстуры с тестовыми данными
│   ├── test_core/               # Тесты реализаций алгоритма
│   │   ├── test_cpu_numpy.py   # Unit-тесты CPU реализации
│   │   └── test_consistency.py # Тесты согласованности реализаций
│   └── test_metrics/            # Тесты метрик и таймеров
│
├── requirements.txt             # Зависимости проекта
├── SETUP.md                     # Инструкции по настройке окружения
└── README.md                    # Документация проекта
```

---

## Эксперименты

Проект включает 6 типов экспериментов для всестороннего анализа производительности.

### Эксперимент 1: Baseline однопоточных реализаций

**Цель**: Установить базовое время выполнения однопоточных реализаций

**Параметры**:
- N = 100,000
- D = 50
- K = 8
- Реализации: Python NumPy
- Повторы: 50

**Запуск**:
```bash
python -m kmeans.main --experiment exp1_baseline_single
```

### Эксперимент 2: Масштабирование по N

**Цель**: Оценить зависимость времени выполнения от размера данных

**Параметры**:
- N ∈ {10³, 10⁵, 10⁶, 5×10⁶}
- D = 50 (фиксировано)
- K = 8 (фиксировано)
- Реализации: CPU (single, multi-process), GPU
- Повторы: 50, 20, 10, 5 соответственно

**Запуск**:
```bash
python -m kmeans.main --experiment exp2_scaling_n
```

### Эксперимент 3: Масштабирование по D

**Цель**: Исследовать влияние размерности признакового пространства на производительность

**Параметры**:
- D ∈ {2, 10, 50, 200}
- N = 100,000 (фиксировано)
- K = 8 (фиксировано)
- Реализации: CPU (single, multi-process), GPU
- Повторы: 20, 20, 10, 10

**Запуск**:
```bash
python -m kmeans.main --experiment exp3_scaling_d
```

### Эксперимент 4: Масштабирование по K

**Цель**: Исследовать влияние числа кластеров на время выполнения

**Параметры**:
- K ∈ {4, 8, 16, 32}
- N = 100,000 (фиксировано)
- D = 50 (фиксировано)
- Реализации: CPU (single, multi-process), GPU
- Повторы: 20, 20, 10, 10

**Запуск**:
```bash
python -m kmeans.main --experiment exp4_scaling_k
```

### Эксперимент 5: Strong Scaling

**Цель**: Анализ масштабируемости по числу процессов/потоков

**Параметры**:
- N = 1,000,000
- D = 50
- K = 8
- Реализации: CPU multiprocessing с различным числом процессов
- Повторы: 15

**Запуск**:
```bash
python -m kmeans.main --experiment exp5_strong_scaling
```

### Эксперимент 6: GPU профилирование

**Цель**: Детальный анализ GPU-реализации для профилирования

**Параметры**:
- N = 1,000,000
- D = 50
- K = 8
- Реализации: GPU (CuPy)
- Повторы: 10

**Запуск**:
```bash
python -m kmeans.main --experiment exp6_gpu_profile
```

**Примечание**: Все эксперименты поддерживают флаг `--gpu-only` для запуска только GPU реализаций (пропуск CPU алгоритмов). Это полезно для быстрого тестирования GPU производительности без ожидания завершения CPU экспериментов.

---

## Реализации алгоритма

### CPU NumPy (Однопоточная)

Базовая реализация на NumPy для последовательных вычислений.

**Использование**:
```python
from kmeans.core.cpu_numpy import KMeansCPUNumpy

model = KMeansCPUNumpy(n_clusters=8, n_iters=100)
model.fit(X, initial_centroids)
```

**Особенности**:
- Прямолинейная реализация для максимальной ясности
- Использует векторные операции NumPy
- Подходит для небольших и средних датасетов
- Сложность одной итерации: O(N × K × D)

### CPU Multiprocessing (Многопроцессная)

Параллельная реализация с использованием `multiprocessing.Pool`.

**Использование**:
```python
from kmeans.core.cpu_multiprocessing import (
    KMeansCPUMultiprocessing,
    MultiprocessingConfig
)

config = MultiprocessingConfig(n_processes=4)
model = KMeansCPUMultiprocessing(n_clusters=8, mp=config)
model.fit(X, initial_centroids)
```

**Особенности**:
- Разделяемая память для данных (RawArray)
- Параллелизация шага назначения кластеров по чанкам объектов
- Параллельная редукция для обновления центроидов
- Переиспользование пула процессов между итерациями
- Оптимизировано для многоядерных систем

### GPU CuPy (CUDA)

Две оптимизированные GPU-реализации на CuPy.

#### Базовая версия

**Использование**:
```python
from kmeans.core.gpu_numpy import KMeansGPUCuPy

model = KMeansGPUCuPy(n_clusters=8)
model.fit(X, initial_centroids)
```

**Особенности**:
- Прямое вычисление расстояний через broadcasting
- Использование `einsum` для оптимизации
- Перенос данных на GPU один раз

#### Версия с оптимизацией через bincount

**Использование**:
```python
from kmeans.core.gpu_numpy import KMeansGPUCuPyBincount

model = KMeansGPUCuPyBincount(n_clusters=8)
model.fit(X, initial_centroids)
```

**Особенности**:
- Оптимизированное вычисление расстояний без материализации полного тензора diff
- Использование формулы: ||x - c||² = ||x||² + ||c||² - 2x·c
- Эффективная редукция через `bincount` для обновления центроидов
- Минимизация использования памяти GPU
- Автоматический возврат результатов на CPU

---

## Анализ результатов

### Формат результатов

Результаты сохраняются в формате NDJSON (каждая строка — отдельный JSON-объект):

```json
{
  "experiment": "exp2_scaling_n",
  "implementation": "python_cpu_numpy",
  "dataset": {
    "N": 100000,
    "D": 50,
    "K": 8,
    "purpose": "scaling_by_N"
  },
  "timing": {
    "T_fit_avg": 20.226,
    "T_fit_std": 0.123,
    "T_fit_min": 20.045,
    "T_assign_total_avg": 18.056,
    "T_update_total_avg": 2.165,
    "T_iter_total_avg": 20.221,
    "runs": [
      {
        "run_idx": 1.0,
        "T_fit": 20.234,
        "T_assign_total": 18.063,
        "T_update_total": 2.171,
        "T_iter_total": 20.234
      }
    ],
    "estimated": false,
    "repeats_done": 50,
    "repeats_requested": 50
  }
}
```

### Статистический анализ

Модуль `analyze_timings.py` предоставляет инструменты для анализа результатов:

```python
from analyze_timings import compute_stats_from_results, TimingResultsAnalyzer

# Простой анализ
compute_stats_from_results("kmeans_timing_results.json", n_iters=100)

# Программный анализ
analyzer = TimingResultsAnalyzer("kmeans_timing_results.json", n_iters=100)
analyzer.analyze()
```

**Вывод включает**:
- Средние и медианные значения времени назначения кластеров (на одну итерацию)
- Средние и медианные значения времени обновления центроидов (на одну итерацию)
- Средние и медианные значения времени одной итерации
- Средние и медианные значения общего времени выполнения алгоритма

### Метрики производительности

Модуль `kmeans/metrics/metrics.py` предоставляет функции для вычисления метрик:

```python
from kmeans.metrics.metrics import speedup, efficiency, throughput

# Ускорение параллельной реализации относительно последовательной
s = speedup(t_serial=100.0, t_parallel=25.0)  # 4.0

# Параллельная эффективность (идеальное значение = 1.0)
e = efficiency(speedup=4.0, p=4)  # 1.0 (линейное ускорение)

# Пропускная способность (операций в секунду)
tp = throughput(N=100000, K=8, D=50, n_iters=100, total_time=20.0)
```

---

## Тестирование

Проект включает комплексный набор тестов для обеспечения корректности работы всех реализаций алгоритма.

### Запуск тестов

```bash
# Все тесты
pytest tests/

# Только тесты согласованности (критично для проверки корректности)
pytest tests/test_core/test_consistency.py -v

# С покрытием кода
pytest tests/ --cov=kmeans --cov-report=html

# Только быстрые тесты (без GPU)
pytest tests/ -m "not gpu"
```

### Структура тестов

- **Тесты согласованности** (`test_consistency.py`) - проверяют, что все реализации дают одинаковые результаты. Критически важно для корректности бенчмарков.
- **Unit-тесты реализаций** (`test_cpu_numpy.py`) - проверяют корректность работы отдельных компонентов алгоритма.
- **Тесты метрик** (`test_metrics/`) - проверяют корректность вычисления метрик производительности.

### Что тестируется

1. **Согласованность реализаций** - CPU NumPy, CPU Multiprocessing, GPU CuPy и GPU CuPy Bincount должны давать одинаковые центроиды на одинаковых данных.
2. **Корректность алгоритма** - проверка шагов `assign_clusters` и `update_centroids`, обработка пустых кластеров.
3. **Метрики производительности** - корректность вычисления speedup, efficiency, throughput.

Подробная информация о стратегии тестирования и примерах тестов доступна в `tests/README.md`.

---

## Расширение проекта

### Добавление новой реализации алгоритма

1. Создайте класс, наследующий `KMeansBase`:

```python
from kmeans.core.base import KMeansBase
import numpy as np

class MyKMeans(KMeansBase):
    """
    Описание вашей реализации.
    """
    
    def assign_clusters(
        self, X: np.ndarray, centroids: np.ndarray
    ) -> np.ndarray:
        """
        Шаг назначения точек кластерам.
        
        Args:
            X: Массив данных (N, D)
            centroids: Текущие центроиды (K, D)
            
        Returns:
            Метки кластеров (N,)
        """
        # Ваша реализация шага назначения
        pass
    
    def update_centroids(
        self, X: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """
        Шаг обновления центроидов.
        
        Args:
            X: Массив данных (N, D)
            labels: Метки кластеров (N,)
            
        Returns:
            Новые центроиды (K, D)
        """
        # Ваша реализация шага обновления
        pass
```

2. Добавьте экспорт в `kmeans/core/__init__.py`:

```python
from .my_kmeans import MyKMeans

__all__ = [..., "MyKMeans"]
```

3. Используйте в экспериментах через фабрику:

```python
def _my_implementation_factory(**kw):
    return MyKMeans(n_clusters=kw['n_clusters'], logger=kw.get('logger'))
```

### Добавление нового эксперимента

1. Добавьте идентификатор в `kmeans/experiments/config.py`:

```python
class ExperimentId(str, Enum):
    # ... существующие эксперименты
    MY_EXPERIMENT = "exp7_my_experiment"
```

2. Добавьте конфигурацию:

```python
EXPERIMENTS: Dict[ExperimentId, ExperimentConfig] = {
    # ... существующие конфигурации
    ExperimentId.MY_EXPERIMENT: ExperimentConfig(
        id=ExperimentId.MY_EXPERIMENT,
        description="Описание эксперимента",
        implementations=["my_impl"],
        params={
            "N": 100000,
            "D": 50,
            "K": 8,
            "repeats": 20,
        },
    ),
}
```

3. Реализуйте метод в `ExperimentSuite`:

```python
def run_exp7_my_experiment(self) -> List[dict]:
    """
    Описание эксперимента.
    
    Returns:
        Список результатов экспериментов
    """
    cfg = EXPERIMENTS[ExperimentId.MY_EXPERIMENT]
    # Ваша логика эксперимента
    results: List[dict] = []
    # ...
    return results
```

4. Добавьте обработку в `kmeans/main.py`:

```python
elif args.experiment == ExperimentId.MY_EXPERIMENT.value:
    suite = make_suite_single()
    results = suite.run_exp7_my_experiment()
```

### Валидация датасетов

Для проверки корректности загрузки датасетов при разработке новых реализаций:

```python
from kmeans.data.validation import validate_dataset
from kmeans.data.dataset import Dataset
from kmeans.data.registry import DatasetRegistry

registry = DatasetRegistry(
    "datasets/datasets_summary.json",
    "datasets"
)

dataset_info = registry.find(N=1000, D=10, K=4)
dataset = Dataset("datasets", dataset_info["metadata"])
validate_dataset(dataset)  # Проверка корректности загрузки
```

---

## Принципы проектирования

Проект следует следующим принципам:

- **Объектно-ориентированное программирование**: Классы с чётким разделением ответственности
- **Типизация**: Полная аннотация типов для всех публичных API
- **Документация**: Подробные docstrings на русском языке
- **Модульность**: Логическое разделение на пакеты и модули
- **Расширяемость**: Легкое добавление новых реализаций и экспериментов
- **Надёжность**: Обработка ошибок и валидация данных

---

## Лицензия

MIT License

---

## Авторы

Проект разработан для исследования производительности алгоритма K-means на различных архитектурах вычислений.

---

## Благодарности

- NumPy и SciPy за эффективные вычисления
- CuPy за GPU-ускорение
- SciKit-learn за генерацию синтетических данных

---

## Дополнительные ресурсы

- [Документация NumPy](https://numpy.org/doc/)
- [Документация CuPy](https://docs.cupy.dev/)
- [Алгоритм K-means (Wikipedia)](https://en.wikipedia.org/wiki/K-means_clustering)
- [Документация pytest](https://docs.pytest.org/)

## Дополнительная документация

- **SETUP.md** - Подробные инструкции по настройке виртуального окружения в PyCharm и других IDE

---

Версия: 1.0.1  
Последнее обновление: 23.12.2025
