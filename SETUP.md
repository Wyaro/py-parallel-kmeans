# Инструкция по настройке окружения

Данная инструкция описывает процесс настройки виртуального окружения для проекта K-Means Benchmark Suite в PyCharm и других IDE.

## Быстрая настройка в PyCharm

### Способ 1: Автоматическая настройка при открытии проекта

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

### Способ 2: Ручная настройка через Settings

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

### Способ 3: Через терминал PyCharm

1. **Откройте терминал в PyCharm**
   - View → Tool Windows → Terminal
   - Или Alt+F12

2. **Создайте виртуальное окружение**
   ```bash
   python -m venv .venv
   ```

3. **Активируйте окружение**
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```

4. **Установите зависимости**
   ```bash
   pip install -r requirements.txt
   ```

5. **Настройте интерпретатор в PyCharm**
   - File → Settings → Project: parallel → Python Interpreter
   - Выберите существующий интерпретатор `.venv\Scripts\python.exe` (Windows)
   - Или `.venv/bin/python` (Linux/macOS)

## Установка GPU поддержки (опционально)

Если у вас есть NVIDIA GPU и установлена CUDA, можно добавить поддержку GPU:

1. **Определите версию CUDA**
   ```bash
   nvidia-smi
   ```
   Или проверьте установленную версию CUDA в системе.

2. **Установите соответствующую версию CuPy**
   
   Для CUDA 13.x:
   ```bash
   pip install cupy-cuda13x
   ```
   
   Для CUDA 12.x:
   ```bash
   pip install cupy-cuda12x
   ```
   
   Для CUDA 11.x:
   ```bash
   pip install cupy-cuda11x
   ```

3. **Проверьте установку**
   ```python
   from kmeans.core.gpu_numpy import gpu_available
   print(f"GPU доступен: {gpu_available()}")
   ```

## Проверка установки

После настройки окружения проверьте, что всё работает:

```python
# Проверка базовых зависимостей
import numpy as np
import sklearn
import matplotlib.pyplot as plt

print(f"NumPy: {np.__version__}")
print(f"SciKit-learn: {sklearn.__version__}")

# Проверка проекта
from kmeans.core.cpu_numpy import KMeansCPUNumpy
from kmeans.core.gpu_numpy import gpu_available

print(f"GPU доступен: {gpu_available()}")
```

## Решение проблем

### Проблема: PyCharm не видит виртуальное окружение

**Решение**:
1. File → Settings → Project: parallel → Python Interpreter
2. Нажмите на шестерёнку → "Add..."
3. Выберите "Existing environment"
4. Укажите путь к `python.exe` в `.venv\Scripts\python.exe` (Windows)

### Проблема: Ошибки при установке зависимостей

**Решение**:
- Убедитесь, что используете Python 3.10 или выше
- Обновите pip: `python -m pip install --upgrade pip`
- Установите зависимости по одной для диагностики:
  ```bash
  pip install numpy
  pip install scikit-learn
  pip install matplotlib
  ```

### Проблема: CuPy не устанавливается

**Решение**:
- CuPy требует CUDA Toolkit. Установите его с официального сайта NVIDIA
- Используйте правильную версию: `cupy-cuda13x` для CUDA 13.x, `cupy-cuda12x` для CUDA 12.x, `cupy-cuda11x` для CUDA 11.x
- GPU поддержка опциональна - проект работает и без неё

## Альтернативные способы установки

### Использование conda

Если вы предпочитаете conda:

```bash
# Создание окружения
conda create -n kmeans-benchmark python=3.10
conda activate kmeans-benchmark

# Установка зависимостей
conda install numpy scipy scikit-learn matplotlib
pip install -r requirements.txt  # Для остальных пакетов
```

### Использование poetry (если добавите pyproject.toml)

```bash
poetry install
poetry shell
```

## Дополнительные настройки PyCharm

### Настройка структуры проекта

1. File → Settings → Project: parallel → Project Structure
2. Убедитесь, что папка `kmeans` помечена как "Sources Root" (синяя)
3. Папки `datasets` и `visualizations` должны быть исключены (если не нужны в индексе)

### Настройка запуска

1. Run → Edit Configurations
2. Добавьте новую конфигурацию Python:
   - Script path: `kmeans/main.py`
   - Parameters: `--experiment exp1_baseline_single`
   - Python interpreter: выберите `.venv`

### Настройка форматирования кода

1. File → Settings → Editor → Code Style → Python
2. Используйте стандартные настройки PEP 8
3. Можно включить автоформатирование при сохранении

---

**Примечание**: Виртуальное окружение `.venv` уже добавлено в `.gitignore` и не будет коммититься в репозиторий. Каждый разработчик создаёт своё локальное окружение.

