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
- Автоматизированные эксперименты: 5 типов экспериментов с различными параметрами
- Генерация датасетов: синтетические данные с контролируемыми параметрами
- Адаптивная остановка: алгоритм останавливается при сходимости (максимум 100 итераций)
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
# Все эксперименты (полный набор для комплексного анализа)
python -m kmeans.main --experiment all

# Конкретный эксперимент
python -m kmeans.main --experiment exp2_scaling_n

# С ограничением времени (30 минут)
python -m kmeans.main --experiment exp2_scaling_n --max-seconds 1800

# Только GPU реализации (пропустить CPU алгоритмы)
python -m kmeans.main --experiment exp2_scaling_n --gpu-only

# Все эксперименты только на GPU
python -m kmeans.main --experiment all --gpu-only
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

**Минимальные требования:**
- **Python**: 3.10 или выше
- **ОС**: Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.15+
- **Память**: минимум 4 GB RAM (рекомендуется 8 GB+)
- **Дисковое пространство**: минимум 1 GB свободного места

**Для GPU поддержки (опционально):**
- **GPU**: NVIDIA GPU с поддержкой CUDA
- **CUDA Toolkit**: версия 11.x, 12.x или 13.x
- **Драйверы NVIDIA**: последние версии драйверов для вашей GPU

### Зависимости

**Основные зависимости:**
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- SciKit-learn >= 1.0.0 (для генерации датасетов)
- Matplotlib >= 3.3.0 (для визуализации)

**Опциональные зависимости:**
- CuPy (для GPU-реализаций, требует CUDA)
  - `cupy-cuda13x` для CUDA 13.x
  - `cupy-cuda12x` для CUDA 12.x
  - `cupy-cuda11x` для CUDA 11.x
- pytest >= 7.0.0 (для тестирования)
- pytest-cov >= 4.0.0 (для анализа покрытия кода)

### Быстрая установка

#### Шаг 1: Клонирование репозитория

```bash
git clone <repository-url>
cd parallel
```

#### Шаг 2: Создание виртуального окружения

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

#### Шаг 3: Установка зависимостей

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Шаг 4: Установка GPU поддержки (опционально)

Определите версию CUDA:
```bash
nvidia-smi
# или
nvcc --version
```

Установите соответствующую версию CuPy:

**Для CUDA 13.x:**
```bash
pip install cupy-cuda13x
```

**Для CUDA 12.x:**
```bash
pip install cupy-cuda12x
```

**Для CUDA 11.x:**
```bash
pip install cupy-cuda11x
```

**Примечание**: Если вы не уверены в версии CUDA, попробуйте установить `cupy-cuda12x` - он обычно совместим с более новыми версиями CUDA.

#### Шаг 5: Генерация датасетов

```bash
python main.py
```

Или только генерация датасетов:
```bash
python -c "from generate_datasets import DatasetGenerator; DatasetGenerator(base_seed=42).generate_all()"
```

### Подробная установка

#### 1. Установка Python

Убедитесь, что у вас установлен Python 3.10 или выше:

```bash
python --version
```

Если Python не установлен, скачайте его с [python.org](https://www.python.org/downloads/).

**Важно**: При установке на Windows убедитесь, что выбрана опция "Add Python to PATH".

#### 2. Создание виртуального окружения

Виртуальное окружение изолирует зависимости проекта от системных пакетов Python.

```bash
# Создание окружения
python -m venv .venv

# Активация (Windows)
.venv\Scripts\activate

# Активация (Linux/macOS)
source .venv/bin/activate
```

После активации в начале строки терминала появится `(.venv)`.

#### 3. Обновление pip

Рекомендуется обновить pip до последней версии:

```bash
python -m pip install --upgrade pip
```

#### 4. Установка основных зависимостей

```bash
pip install -r requirements.txt
```

Это установит все необходимые пакеты из `requirements.txt`.

#### 5. Проверка установки основных зависимостей

```python
import numpy as np
import scipy
import sklearn
import matplotlib

print(f"NumPy: {np.__version__}")
print(f"SciPy: {scipy.__version__}")
print(f"SciKit-learn: {sklearn.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
```

### Настройка в PyCharm и других IDE

#### Способ 1: Автоматическая настройка при открытии проекта

1. **Откройте проект в PyCharm**
   - File → Open → выберите папку `parallel`

2. **PyCharm автоматически предложит настроить интерпретатор**
   - Внизу появится уведомление "No Python interpreter configured"
   - Нажмите "Configure Python Interpreter"

3. **Создайте новое виртуальное окружение**
   - Выберите "New environment"
   - Location: `.venv` (в корне проекта)
   - Base interpreter: выберите Python 3.10 или выше
   - Убедитесь, что галочка "Inherit global site-packages" снята
   - Нажмите "OK"

4. **Установите зависимости**
   - PyCharm автоматически обнаружит `requirements.txt`
   - Появится уведомление "Requirements file 'requirements.txt' detected"
   - Нажмите "Install requirements"

#### Способ 2: Ручная настройка через Settings

1. **Откройте настройки интерпретатора**
   - File → Settings (Ctrl+Alt+S)
   - Project: parallel → Python Interpreter

2. **Добавьте новый интерпретатор**
   - Нажмите на шестерёнку рядом с интерпретатором
   - Выберите "Add..."

3. **Создайте виртуальное окружение**
   - Выберите "New environment"
   - Location: `.venv`
   - Base interpreter: Python 3.10+
   - Нажмите "OK"

4. **Установите зависимости**
   - В окне Python Interpreter нажмите кнопку "+" (Install packages)
   - Или выполните в терминале PyCharm:
     ```bash
     pip install -r requirements.txt
     ```

#### Способ 3: Через терминал IDE

1. **Откройте терминал в IDE**
   - PyCharm: View → Tool Windows → Terminal (или Alt+F12)

2. **Создайте виртуальное окружение**
   ```bash
   python -m venv .venv
   ```

3. **Активируйте окружение**
   - Windows: `.venv\Scripts\activate`
   - Linux/macOS: `source .venv/bin/activate`

4. **Установите зависимости**
   ```bash
   pip install -r requirements.txt
   ```

5. **Настройте интерпретатор в IDE**
   - File → Settings → Project: parallel → Python Interpreter
   - Выберите существующий интерпретатор `.venv\Scripts\python.exe` (Windows) или `.venv/bin/python` (Linux/macOS)

#### Дополнительные настройки PyCharm

**Настройка структуры проекта:**
1. File → Settings → Project: parallel → Project Structure
2. Убедитесь, что папка `kmeans` помечена как "Sources Root" (синяя)
3. Папки `datasets` и `visualizations` могут быть исключены (если не нужны в индексе)

**Настройка запуска:**
1. Run → Edit Configurations
2. Добавьте новую конфигурацию Python:
   - Script path: `kmeans/main.py`
   - Parameters: `--experiment exp1_baseline_single`
   - Python interpreter: выберите `.venv`

### Проверка установки

#### Базовая проверка

```python
# Проверка базовых зависимостей
import numpy as np
import sklearn
import matplotlib.pyplot as plt

print(f"NumPy версия: {np.__version__}")
print(f"SciKit-learn версия: {sklearn.__version__}")

# Проверка проекта
from kmeans.core.cpu_numpy import KMeansCPUNumpy
from kmeans.core.gpu_numpy import gpu_available

print(f"GPU доступен: {gpu_available()}")
```

#### Полная проверка установки

Создайте файл `check_installation.py`:

```python
"""Проверка установки K-Means Benchmark Suite."""

import sys

def check_python_version():
    """Проверка версии Python."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (требуется 3.10+)")
        return False

def check_dependencies():
    """Проверка основных зависимостей."""
    dependencies = {
        "numpy": "NumPy",
        "scipy": "SciPy",
        "sklearn": "SciKit-learn",
        "matplotlib": "Matplotlib",
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name}: не установлен")
            all_ok = False
    
    return all_ok

def check_gpu():
    """Проверка GPU поддержки."""
    try:
        from kmeans.core.gpu_numpy import gpu_available
        if gpu_available():
            import cupy as cp
            print(f"✓ GPU доступен (CuPy {cp.__version__})")
            return True
        else:
            print("⚠ GPU недоступен (опционально)")
            return True  # Не критично
    except ImportError:
        print("⚠ GPU поддержка не установлена (опционально)")
        return True  # Не критично

def check_project_modules():
    """Проверка модулей проекта."""
    modules = [
        "kmeans.core.cpu_numpy",
        "kmeans.core.cpu_multiprocessing",
        "kmeans.data.dataset",
        "kmeans.experiments.suite",
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            all_ok = False
    
    return all_ok

if __name__ == "__main__":
    print("Проверка установки K-Means Benchmark Suite\n")
    print("=" * 50)
    
    checks = [
        ("Версия Python", check_python_version),
        ("Зависимости", check_dependencies),
        ("Модули проекта", check_project_modules),
        ("GPU поддержка", check_gpu),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        results.append(check_func())
    
    print("\n" + "=" * 50)
    if all(results[:3]):  # Первые 3 проверки обязательны
        print("✓ Установка успешна!")
    else:
        print("✗ Обнаружены проблемы. Проверьте ошибки выше.")
        sys.exit(1)
```

Запустите проверку:
```bash
python check_installation.py
```

### Решение проблем

#### Проблема: Python не найден

**Симптомы**: `python: command not found` или `'python' is not recognized`

**Решение**:
1. Убедитесь, что Python установлен: `python --version`
2. На Windows добавьте Python в PATH
3. Попробуйте использовать `python3` вместо `python` (Linux/macOS)

#### Проблема: Ошибки при установке зависимостей

**Симптомы**: `ERROR: Could not find a version that satisfies the requirement`

**Решение**:
1. Обновите pip: `python -m pip install --upgrade pip`
2. Установите зависимости по одной для диагностики:
   ```bash
   pip install numpy
   pip install scipy
   pip install scikit-learn
   pip install matplotlib
   ```
3. На Windows может потребоваться установка Visual C++ Build Tools

#### Проблема: CuPy не устанавливается

**Симптомы**: Ошибки при установке `cupy-cuda*`

**Решение**:
1. Убедитесь, что CUDA Toolkit установлен
2. Проверьте версию CUDA: `nvidia-smi` или `nvcc --version`
3. Установите правильную версию CuPy для вашей CUDA
4. GPU поддержка опциональна - проект работает и без неё

#### Проблема: Ошибки импорта модулей проекта

**Симптомы**: `ModuleNotFoundError: No module named 'kmeans'`

**Решение**:
1. Убедитесь, что вы находитесь в корневой директории проекта
2. Проверьте, что виртуальное окружение активировано
3. Установите зависимости: `pip install -r requirements.txt`

#### Проблема: PyCharm не видит виртуальное окружение

**Решение**:
1. File → Settings → Project: parallel → Python Interpreter
2. Нажмите на шестерёнку → "Add..."
3. Выберите "Existing environment"
4. Укажите путь к `python.exe` в `.venv\Scripts\python.exe` (Windows)

#### Проблема: Недостаточно памяти при генерации датасетов

**Симптомы**: `MemoryError` при генерации больших датасетов

**Решение**:
1. Генератор автоматически ограничивает размер больших датасетов
2. Для очень больших N (>10M) датасет будет ограничен до 5M точек
3. Убедитесь, что у вас достаточно свободной RAM

#### Проблема: GPU недоступен после установки CuPy

**Симптомы**: `gpu_available()` возвращает `False`

**Решение**:
1. Проверьте, что GPU видна системой: `nvidia-smi`
2. Убедитесь, что установлена правильная версия CuPy
3. Проверьте совместимость CUDA и CuPy версий
4. Перезапустите Python после установки CuPy

### Альтернативные способы установки

#### Использование conda

```bash
# Создание окружения
conda create -n kmeans-benchmark python=3.10
conda activate kmeans-benchmark

# Установка зависимостей
conda install numpy scipy scikit-learn matplotlib
pip install -r requirements.txt  # Для остальных пакетов

# Для GPU (если доступно)
conda install -c conda-forge cupy
```

**Примечание**: Виртуальное окружение `.venv` уже добавлено в `.gitignore` и не будет коммититься в репозиторий. Каждый разработчик создаёт своё локальное окружение.

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
# Все эксперименты (рекомендуется для полного анализа)
python -m kmeans.main --experiment all

# Базовый эксперимент (однопоточная реализация)
python -m kmeans.main --experiment exp1_baseline_single

# Масштабирование по количеству точек
python -m kmeans.main --experiment exp2_scaling_n

# Масштабирование по размерности
python -m kmeans.main --experiment exp3_scaling_d

# Масштабирование по количеству кластеров
python -m kmeans.main --experiment exp4_scaling_k

# GPU профилирование
python -m kmeans.main --experiment exp5_gpu_profile
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
model = KMeansCPUNumpy(n_clusters=8, n_iters=100, tol=1e-6)
model.fit(X, initial_centroids)
print(f"Центроиды: {model.centroids.shape}")
print(f"Метки: {model.labels.shape}")
print(f"Выполнено итераций: {model.n_iters_actual}")

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
└── README.md                    # Документация проекта
```

---

## Эксперименты

Проект включает 5 типов экспериментов для всестороннего анализа производительности.

### Запуск всех экспериментов

Для запуска всех экспериментов последовательно используйте команду:

```bash
python -m kmeans.main --experiment all
```

Эта команда запустит все 5 типов экспериментов с соответствующими параметрами и реализациями. Результаты будут сохраняться в `kmeans_timing_results.json` по мере выполнения.

**Дополнительные опции:**

```bash
# Все эксперименты с ограничением времени (30 минут)
python -m kmeans.main --experiment all --max-seconds 1800

# Все эксперименты только на GPU (пропустить CPU алгоритмы)
python -m kmeans.main --experiment all --gpu-only
```

### Эксперимент 1: Baseline однопоточных реализаций

**Цель**: Установить базовое время выполнения однопоточных реализаций

**Параметры**:
- N = 100,000
- D = 50
- K = 8
- Реализации: Python NumPy
- Повторы: 30 (warmup: 3)

**Запуск**:
```bash
python -m kmeans.main --experiment exp1_baseline_single
```

### Эксперимент 2: Масштабирование по N

**Цель**: Оценить зависимость времени выполнения от размера данных

**Параметры**:
- N ∈ {1,000, 100,000, 1,000,000, 5,000,000}
- D = 50 (фиксировано)
- K = 8 (фиксировано)
- Реализации: CPU (single, multi-process), GPU
- Повторы: 50, 20, 10, 5 соответственно (для каждого N)

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
- Повторы: 20, 20, 10, 10 соответственно (для каждого D)

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
- Повторы: 20, 20, 10, 10 соответственно (для каждого K)

**Запуск**:
```bash
python -m kmeans.main --experiment exp4_scaling_k
```

### Эксперимент 5: GPU профилирование

**Цель**: Детальный анализ GPU-реализации для профилирования

**Параметры**:
- N = 1,000,000
- D = 50
- K = 8
- Реализации: GPU (CuPy)
- Повторы: 10

**Запуск**:
```bash
python -m kmeans.main --experiment exp5_gpu_profile
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
- Адаптивная остановка: останавливается при сходимости (tol=1e-6) или максимум 100 итераций

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
- Адаптивная остановка: останавливается при сходимости (tol=1e-6) или максимум 100 итераций

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
- Адаптивная остановка: останавливается при сходимости (tol=1e-6) или максимум 100 итераций

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
- Адаптивная остановка: останавливается при сходимости (tol=1e-6) или максимум 100 итераций

---

## Методология реализации

Все реализации алгоритма K-means следуют единой архитектуре, основанной на абстрактном базовом классе `KMeansBase`, который определяет общий алгоритм итераций и сбор метрик производительности. Конкретные реализации различаются подходом к параллелизации и оптимизации ключевых операций.

### Общая архитектура

Все реализации наследуются от базового класса `KMeansBase`, который обеспечивает:

1. **Единый цикл итераций**: Метод `fit()` реализует основной алгоритм с адаптивной остановкой по сходимости
2. **Сбор метрик**: Автоматическое измерение времени выполнения шагов `assign_clusters` и `update_centroids`
3. **Проверка сходимости**: Остановка алгоритма при достижении порога изменения центроидов (`tol`) или максимального числа итераций

Ключевые абстрактные методы, которые должны быть реализованы в каждой версии:
- `assign_clusters(X, centroids)` — назначение точек кластерам
- `update_centroids(X, labels)` — обновление центроидов по меткам

### Однопоточная CPU реализация

**Принципы реализации:**

1. **Последовательное выполнение**: Все вычисления выполняются в одном потоке с использованием векторных операций NumPy
2. **Векторизация**: Использование broadcasting и матричных операций NumPy для эффективных вычислений
3. **Простота**: Минимальные накладные расходы, максимальная читаемость кода

**Алгоритм шага назначения (`assign_clusters`):**
- Вычисление разности: `diff = X[:, None, :] - centroids[None, :, :]` (broadcasting до формы (N, K, D))
- Вычисление квадратов расстояний: `distances = np.sum(diff * diff, axis=2)` (результат: (N, K))
- Поиск ближайшего кластера: `labels = np.argmin(distances, axis=1)` (результат: (N,))

**Алгоритм шага обновления (`update_centroids`):**
- Для каждого кластера k:
  - Выбор точек кластера: `points = X[labels == k]`
  - Вычисление среднего: `centroids[k] = points.mean(axis=0)`
- Обработка пустых кластеров: сохранение предыдущих координат центроида

**Сложность:**
- Время: O(N × K × D) на итерацию
- Память: O(N × K × D) для временного массива diff

### Многопроцессная CPU реализация

**Принципы реализации:**

1. **Разделяемая память**: Использование `multiprocessing.RawArray` для избежания копирования данных между процессами
2. **Параллелизация по данным**: Разбиение набора точек на чанки, обработка каждого чанка в отдельном процессе
3. **Переиспользование ресурсов**: Пул процессов создаётся один раз и переиспользуется между итерациями
4. **Параллельная редукция**: Агрегация результатов от всех процессов для обновления центроидов

**Архитектура параллелизации:**

**Шаг назначения (`assign_clusters`):**
1. Разбиение индексов на чанки: `chunks = np.array_split(np.arange(N), n_processes)`
2. Параллельное выполнение в воркерах:
   - Каждый воркер получает индексы своего чанка и центроиды
   - Чтение данных из разделяемой памяти (`RawArray`)
   - Вычисление меток для точек чанка
   - Возврат меток чанка
3. Сборка результатов: объединение меток от всех воркеров в единый массив

**Шаг обновления (`update_centroids`):**
1. Параллельная частичная редукция:
   - Каждый воркер обрабатывает свой чанк точек
   - Вычисляет частичные суммы и счётчики для каждого кластера: `sums[k]`, `counts[k]`
   - Возвращает пару `(sums, counts)` размером (K, D) и (K,)
2. Глобальная редукция:
   - Суммирование частичных результатов от всех воркеров
   - Вычисление новых центроидов: `centroids[k] = sums[k] / counts[k]`

**Оптимизации:**
- Ленивая инициализация пула: создание происходит только при первом вызове
- Фиксированное разбиение на чанки: переиспользование структуры между итерациями
- Минимизация передачи данных: только индексы и центроиды передаются в воркеры

**Сложность:**
- Время: O(N × K × D / P) на итерацию, где P — число процессов
- Память: O(N × D) для разделяемых данных + O(K × D × P) для временных результатов

### Многопоточная GPU реализация

**Принципы реализации:**

1. **Единоразовый перенос данных**: Данные переносятся на GPU один раз в начале `fit()`, остаются на GPU в течение всех итераций
2. **Векторизация на GPU**: Использование массивов CuPy и CUDA-кернелов для параллельного выполнения операций
3. **Оптимизация памяти**: Минимизация использования глобальной памяти GPU через оптимизированные алгоритмы
4. **Асинхронное выполнение**: Операции на GPU выполняются асинхронно относительно CPU

**Архитектура выполнения:**

**Инициализация (`fit`):**
1. Перенос данных на GPU: `X_gpu = cp.asarray(X)`, `centroids_gpu = cp.asarray(initial_centroids)`
2. Измерение времени передачи Host-to-Device (H2D)
3. Все последующие вычисления выполняются на GPU

**Шаг назначения (`assign_clusters`):**

*Вариант 1: Базовая реализация (KMeansGPUCuPy)*
- Вычисление разности через broadcasting: `diff = X[:, None, :] - centroids[None, :, :]`
- Вычисление расстояний через `einsum`: `distances = cp.einsum("nkd,nkd->nk", diff, diff)`
- Поиск минимума: `labels = cp.argmin(distances, axis=1)`
- Память: O(N × K × D) для временного массива diff

*Вариант 2: Оптимизированная реализация (KMeansGPUCuPyBincount)*
- Использование матричной формулы без материализации diff:
  - `distances = ||X||² + ||centroids||² - 2 × X @ centroids.T`
- Вычисление через матричное умножение: `cross = X @ centroids.T`
- Память: O(N × K) вместо O(N × K × D)

*Вариант 3: Raw CUDA кернел (KMeansGPUCuPyRaw)*
- Каждый CUDA-поток обрабатывает одну точку
- Внутри потока: последовательный перебор всех кластеров для поиска ближайшего
- Минимизация использования глобальной памяти через локальные переменные

**Шаг обновления (`update_centroids`):**

*Вариант 1: Scatter-add (KMeansGPUCuPy)*
- Использование атомарных операций: `cp.add.at(sums, labels, X)`
- Параллельное накопление сумм и счётчиков для всех точек одновременно
- Простота реализации, но возможны конфликты при параллельном доступе

*Вариант 2: Bincount редукция (KMeansGPUCuPyBincount)*
- Для каждого признака d:
  - `sums[:, d] = cp.bincount(labels, weights=X[:, d], minlength=K)`
- Последовательная обработка признаков, но эффективная редукция внутри каждого
- Меньше конфликтов памяти, лучше масштабируемость

**Завершение (`fit`):**
1. Перенос результатов на CPU: `centroids_cpu = cp.asnumpy(centroids_gpu)`
2. Измерение времени передачи Device-to-Host (D2H)

**Оптимизации:**
- Использование float32 вместо float64 для уменьшения использования памяти и пропускной способности
- Оптимизация формул вычисления расстояний для минимизации операций
- Переиспользование GPU-массивов между итерациями

**Сложность:**
- Время: O(N × K × D / T) на итерацию, где T — число потоков GPU (тысячи)
- Память: O(N × D) для данных + O(N × K) или O(N × K × D) для временных массивов (в зависимости от варианта)

### Сравнение подходов

| Аспект | Однопоточная CPU | Многопроцессная CPU | GPU |
|--------|------------------|---------------------|-----|
| **Параллелизм** | Последовательное выполнение | Параллелизм по данным (процессы) | Массовый параллелизм (потоки) |
| **Управление памятью** | Единое адресное пространство | Разделяемая память (RawArray) | Отдельная память GPU |
| **Передача данных** | Нет накладных расходов | Минимальные (только индексы) | Значительные (H2D/D2H) |
| **Масштабируемость** | Линейная по N | Линейная по P (процессам) | Экспоненциальная по T (потокам) |
| **Накладные расходы** | Минимальные | Создание/синхронизация процессов | Перенос данных, синхронизация потоков |
| **Оптимальный случай** | Малые/средние датасеты | Средние/большие датасеты, многоядерные CPU | Очень большие датасеты, высокая размерность |

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


## Авторы
О-21-ИАС-аид-С
Еремин И. А.
Проект разработан для исследования производительности алгоритма K-means на различных архитектурах вычислений.

---

## Дополнительные ресурсы

- [Документация NumPy](https://numpy.org/doc/)
- [Документация CuPy](https://docs.cupy.dev/)
- [Алгоритм K-means (Wikipedia)](https://en.wikipedia.org/wiki/K-means_clustering)
- [Документация pytest](https://docs.pytest.org/)

## Дополнительная документация

Вся информация по установке и настройке проекта находится в разделе [Установка и требования](#установка-и-требования) выше.

---

Версия: 1.0.2
Последнее обновление: 23.12.2025
