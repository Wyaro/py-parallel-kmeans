import numpy as np
import json
import os
from pathlib import Path
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import time


class DatasetGenerator:
    def __init__(self, base_seed=42):
        self.base_seed = base_seed
        self.datasets_dir = Path("datasets")
        self.create_directory_structure()

    def create_directory_structure(self):
        directories = [
            "base",
            "scaling_N",
            "scaling_D",
            "scaling_K",
            "validation"
        ]

        for dir_name in directories:
            (self.datasets_dir / dir_name).mkdir(parents=True, exist_ok=True)

    def generate_blobs_dataset(self, N, D, K, cluster_std=1.0, seed_offset=0):
        """
        Генерация синтетического датасета с помощью make_blobs

        Args:
            N: количество точек
            D: размерность
            K: количество кластеров
            cluster_std: стандартное отклонение кластеров
            seed_offset: смещение seed для разных конфигураций

        Returns:
            data: массив данных (N x D)
            labels: метки кластеров (N,)
            centers: центры кластеров (K x D)
        """

        seed = self.base_seed
        np.random.seed(seed)

        print(f"Генерация: N={N:,}, D={D}, K={K}, seed={seed}")


        data, labels, centers = make_blobs(
            n_samples=N,
            n_features=D,
            centers=K,
            cluster_std=cluster_std,
            center_box=(-10.0, 10.0),
            random_state=seed,
            return_centers=True
        )

        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        centers = scaler.transform(centers)

        return data, labels, centers

    def save_dataset_txt(self, data, labels, centers, filepath, metadata):
        """
        Сохранение датасета в текстовом формате

        Формат файла:
        # Метаданные в формате JSON
        # Центроиды (K строк, D+1 колонок: метка + координаты)
        # Данные (N строк, D+1 колонок: метка + координаты)
        """
        filepath = Path(filepath)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# " + json.dumps(metadata, ensure_ascii=False) + "\n")
            f.write("\n")

            f.write("# Centroids (label, x1, x2, ..., xD)\n")
            for k in range(len(centers)):
                centroid_line = f"{k} " + " ".join([f"{coord:.8f}" for coord in centers[k]])
                f.write(centroid_line + "\n")
            f.write("\n")

            f.write("# Data points (label, x1, x2, ..., xD)\n")
            for i in range(len(data)):
                point_line = f"{labels[i]} " + " ".join([f"{coord:.8f}" for coord in data[i]])
                f.write(point_line + "\n")

        metadata_path = filepath.parent / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                existing_metadata = json.load(f)
        else:
            existing_metadata = {}

        existing_metadata[filepath.stem] = metadata
        with open(metadata_path, 'w') as f:
            json.dump(existing_metadata, f, indent=2)

        print(f"  Сохранено: {filepath} ({data.shape[0]:,} точек, {data.shape[1]}D)")

    def generate_base_configurations(self):
        print("\n" + "=" * 60)
        print("Генерация базовых конфигураций")
        print("=" * 60)

        base_config = {
            'N': 100000,
            'D': 50,
            'K': 8,
            'cluster_std': 1.0
        }

        data, labels, centers = self.generate_blobs_dataset(**base_config)

        metadata = {
            **base_config,
            'generated': time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_points': len(data),
            'dimensions': data.shape[1],
            'n_clusters': len(centers),
            'data_type': 'synthetic_blobs',
            'normalized': True,
            'seed': self.base_seed
        }

        filename = f"N{base_config['N']}_D{base_config['D']}_K{base_config['K']}.txt"
        filepath = self.datasets_dir / "base" / filename

        self.save_dataset_txt(data, labels, centers, filepath, metadata)

    def generate_scaling_N(self):
        print("\n" + "=" * 60)
        print("Генерация датасетов: масштабирование по N")
        print("=" * 60)

        N_values = [1000, 100000, 1000000, 1000000000]  # 10^3, 10^5, 10^6, 10^9

        for i, N in enumerate(N_values):
            # Для больших N используем экономную генерацию
            if N > 10_000_000:
                print(f"ВНИМАНИЕ: N={N:,} слишком велико для памяти. Генерируем уменьшенную версию.")
                N = 5_000_000  # Практический предел для большинства систем

            config = {
                'N': N,
                'D': 50,
                'K': 8,
                'cluster_std': 1.0,
                'seed_offset': i
            }

            data, labels, centers = self.generate_blobs_dataset(**config)

            metadata = {
                **config,
                'generated': time.strftime("%Y-%m-%d %H:%M:%S"),
                'total_points': len(data),
                'dimensions': data.shape[1],
                'n_clusters': len(centers),
                'data_type': 'synthetic_blobs',
                'normalized': True,
                'purpose': 'scaling_by_N'
            }

            filename = f"N{N}_D{config['D']}_K{config['K']}.txt"
            filepath = self.datasets_dir / "scaling_N" / filename

            self.save_dataset_txt(data, labels, centers, filepath, metadata)

    def generate_scaling_D(self):
        print("\n" + "=" * 60)
        print("Генерация датасетов: масштабирование по D")
        print("=" * 60)

        D_values = [2, 10, 50, 200]

        for i, D in enumerate(D_values):
            config = {
                'N': 100000,
                'D': D,
                'K': 8,
                'cluster_std': 1.0,
                'seed_offset': i + 10
            }

            data, labels, centers = self.generate_blobs_dataset(**config)

            metadata = {
                **config,
                'generated': time.strftime("%Y-%m-%d %H:%M:%S"),
                'total_points': len(data),
                'dimensions': data.shape[1],
                'n_clusters': len(centers),
                'data_type': 'synthetic_blobs',
                'normalized': True,
                'purpose': 'scaling_by_D'
            }

            filename = f"N{config['N']}_D{D}_K{config['K']}.txt"
            filepath = self.datasets_dir / "scaling_D" / filename

            self.save_dataset_txt(data, labels, centers, filepath, metadata)

    def generate_scaling_K(self):
        print("\n" + "=" * 60)
        print("Генерация датасетов: масштабирование по K")
        print("=" * 60)

        K_values = [4, 8, 16, 32]

        for i, K in enumerate(K_values):
            config = {
                'N': 100000,
                'D': 50,
                'K': K,
                'cluster_std': 1.0,
                'seed_offset': i + 20
            }

            data, labels, centers = self.generate_blobs_dataset(**config)

            metadata = {
                **config,
                'generated': time.strftime("%Y-%m-%d %H:%M:%S"),
                'total_points': len(data),
                'dimensions': data.shape[1],
                'n_clusters': len(centers),
                'data_type': 'synthetic_blobs',
                'normalized': True,
                'purpose': 'scaling_by_K'
            }

            filename = f"N{config['N']}_D{config['D']}_K{K}.txt"
            filepath = self.datasets_dir / "scaling_K" / filename

            self.save_dataset_txt(data, labels, centers, filepath, metadata)

    def generate_validation_data(self):
        print("\n" + "=" * 60)
        print("Подготовка данных для валидации")
        print("=" * 60)

        # Пример: маленький датасет для быстрой проверки
        config = {
            'N': 1000,
            'D': 10,
            'K': 4,
            'cluster_std': 0.5,
            'seed_offset': 100
        }

        data, labels, centers = self.generate_blobs_dataset(**config)

        metadata = {
            **config,
            'generated': time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_points': len(data),
            'dimensions': data.shape[1],
            'n_clusters': len(centers),
            'data_type': 'synthetic_blobs',
            'normalized': True,
            'purpose': 'validation'
        }

        filename = "validation_dataset.txt"
        filepath = self.datasets_dir / "validation" / filename

        self.save_dataset_txt(data, labels, centers, filepath, metadata)


    def generate_all(self):
        print("Начало генерации синтетических датасетов")
        print(f"Базовый seed: {self.base_seed}")
        print(f"Время начала: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        start_time = time.time()

        self.generate_base_configurations()
        self.generate_scaling_N()
        self.generate_scaling_D()
        self.generate_scaling_K()
        self.generate_validation_data()

        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Генерация завершена за {elapsed:.2f} секунд")
        print(f"Датасеты сохранены в: {self.datasets_dir.absolute()}")

        # Создание summary
        self.create_summary()

    def create_summary(self):
        summary = {
            "project": "K-means Benchmark Datasets",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_seed": self.base_seed,
            "total_datasets": 0,
            "datasets": []
        }

        # Собираем информацию о всех датасетах
        for root, dirs, files in os.walk(self.datasets_dir):
            for file in files:
                if file.endswith('.txt') and not file.startswith('.'):
                    filepath = Path(root) / file
                    rel_path = filepath.relative_to(self.datasets_dir)

                    # Читаем первую строку с метаданными
                    with open(filepath, 'r') as f:
                        first_line = f.readline().strip('# \n')

                    try:
                        metadata = json.loads(first_line)
                        metadata['filepath'] = str(rel_path)
                        metadata['size_mb'] = os.path.getsize(filepath) / (1024 * 1024)
                        summary["datasets"].append(metadata)
                        summary["total_datasets"] += 1
                    except:
                        pass

        # Сохраняем summary
        summary_path = self.datasets_dir / "datasets_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Сводный отчет: {summary_path}")

        # Выводим статистику
        print("\nСтатистика датасетов:")
        categories = {}
        for ds in summary["datasets"]:
            cat = ds.get('purpose', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1

        for cat, count in categories.items():
            print(f"  {cat}: {count} датасетов")


