# K-Means Benchmark Suite (C++/CUDA)

> Профессиональная C++ реализация исследовательской платформы для сравнения эффективности K-means на CPU и GPU (CUDA).

## Описание

Проект повторяет архитектуру Python-версии и обеспечивает:

- однопоточную CPU реализацию
- многопоточную CPU реализацию (OpenMP)
- GPU реализацию (CUDA)
- запуск экспериментов по масштабированию
- сбор метрик производительности в формате NDJSON

## Быстрый старт

### Требования

- CMake 3.22+
- Компилятор C++17
- NVIDIA CUDA Toolkit (для GPU-реализации)
- OpenMP (опционально)

### Сборка

```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j
```

### Запуск

```bash
./kmeans_benchmark --experiment exp2_scaling_n
```

Доступные эксперименты:

- `exp2_scaling_n` — масштабирование по N
- `exp3_scaling_d` — масштабирование по D
- `exp4_scaling_k` — масштабирование по K
- `all` — полный запуск

Результаты сохраняются в `kmeans_timing_results.json`.

## Структура проекта

```
cpp_cuda_kmeans/
├── CMakeLists.txt
├── README.md
├── scripts/
│   └── generate_datasets.cpp
└── src/
    ├── main.cpp
    ├── core/
    ├── data/
    ├── experiments/
    ├── metrics/
    └── utils/
```

