"""
Визуализация синтетических датасетов для бенчмаркинга алгоритма K-means.

Модуль содержит функции для:
- загрузки датасетов, сгенерированных `generate_datasets.py`;
- построения 2D / 3D визуализаций;
- построения статистических графиков для высокомерных данных;
- пакетной обработки всех датасетов и формирования HTML-отчета.

Основная точка входа для пакетной обработки — функция `visualize_all_datasets`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence, Tuple

import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


@dataclass(frozen=True)
class DatasetPaths:
    """Набор путей, используемых при визуализации датасетов."""

    datasets_dir: Path = Path("datasets")
    output_dir: Path = Path("visualizations")


def load_dataset(
    filepath: str | Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Загрузка датасета из текстового файла.

    Формат:
    # метаданные в JSON
    # Центроиды
    0 x1 x2 ...
    1 x1 x2 ...
    # Точки данных
    0 x1 x2 ...
    1 x1 x2 ...
    """
    filepath = Path(filepath)

    with filepath.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # Парсинг метаданных
    metadata_line = lines[0].strip("# \n")
    metadata = json.loads(metadata_line)

    # Поиск секций
    centroids_start = None
    data_start = None

    for i, line in enumerate(lines):
        if "Centroids" in line:
            centroids_start = i + 1
        elif "Data points" in line:
            data_start = i + 1

    # Парсинг центроидов
    centroids = []
    for i in range(centroids_start, data_start - 2):
        line = lines[i].strip()
        if line:
            parts = list(map(float, line.split()))
            centroids.append(parts[1:])

    # Парсинг точек данных
    data = []
    labels = []
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if line:
            parts = list(map(float, line.split()))
            labels.append(int(parts[0]))
            data.append(parts[1:])

    return np.asarray(data), np.asarray(labels), np.asarray(centroids), metadata


def configure_plot_style() -> None:
    """Настройка стиля графиков."""
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "figure.titleweight": "bold",
        }
    )


def create_info_box(
    ax: Any,
    text: str,
    position: str = "top_left",
    bg_color: str = "#E8F4F8",
    edge_color: str = "#3498DB",
) -> None:
    """Создание информационного блока на графике."""
    if position == "top_left":
        x, y, va, ha = 0.02, 0.98, "top", "left"
    elif position == "top_right":
        x, y, va, ha = 0.98, 0.98, "top", "right"
    else:  # bottom_left
        x, y, va, ha = 0.02, 0.02, "bottom", "left"

    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        verticalalignment=va,
        horizontalalignment=ha,
        fontsize=9,
        fontweight="bold",
        bbox=dict(
            boxstyle="round",
            facecolor=bg_color,
            alpha=0.9,
            edgecolor=edge_color,
            linewidth=2,
        ),
    )


def add_ellipses(
    ax: Any,
    data: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> None:
    """Добавление эллипсов для визуализации распределения кластеров."""
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = data[labels == label]
        if len(cluster_points) > 1:
            cov = np.cov(cluster_points.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigvals)

            ellipse = Ellipse(
                xy=centroids[label],
                width=width,
                height=height,
                angle=angle,
                alpha=0.15,
                color=plt.cm.tab20c(label / max(1, len(unique_labels) - 1)),
                linestyle="--",
                linewidth=1,
                edgecolor="gray",
            )
            ax.add_patch(ellipse)


def plot_dataset_2d(
    data: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    metadata: dict[str, Any],
    save_path: Path,
) -> None:
    """2D визуализация датасета."""
    configure_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("2D Визуализация кластеров", y=1.02)

    # Подготовка данных для 2D отображения
    if data.shape[1] >= 2:
        plot_data = data[:, :2]
        plot_centroids = centroids[:, :2] if centroids.shape[1] >= 2 else centroids
        x_label, y_label = "Признак 1", "Признак 2"
    else:
        plot_data = np.column_stack([data[:, 0], np.zeros(len(data))])
        plot_centroids = np.column_stack([centroids[:, 0], np.zeros(len(centroids))])
        x_label, y_label = "Признак", ""

    # График 1: Scatter plot
    ax1 = axes[0]
    ax1.scatter(
        plot_data[:, 0],
        plot_data[:, 1],
        c=labels,
        cmap="tab20c",
        alpha=0.7,
        s=30,
        edgecolors="white",
        linewidth=0.5,
    )

    ax1.scatter(
        plot_centroids[:, 0],
        plot_centroids[:, 1],
        c="#FF6B6B",
        marker="*",
        s=350,
        edgecolors="black",
        linewidth=2,
        label="Центроиды",
        zorder=10,
    )

    add_ellipses(ax1, plot_data, labels, plot_centroids)

    ax1.set_xlabel(x_label, fontweight="bold")
    ax1.set_ylabel(y_label, fontweight="bold")
    ax1.set_title("Распределение точек и центроидов", pad=15)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.4, linestyle="--")

    # Информация о датасете
    info_text = (
        f"Параметры датасета:\n"
        f"Точки (N): {metadata.get('N', len(data)):,}\n"
        f"Признаки (D): {metadata.get('D', data.shape[1])}\n"
        f"Кластеры (K): {metadata.get('K', len(centroids))}"
    )
    create_info_box(ax1, info_text)

    # График 2: Гистограмма размеров кластеров
    ax2 = axes[1]
    unique_labels, counts = np.unique(labels, return_counts=True)
    colors = plt.cm.tab20c(np.linspace(0, 1, len(unique_labels)))

    bars = ax2.bar(
        unique_labels,
        counts,
        color=colors,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )
    ax2.set_xlabel("ID кластера", fontweight="bold")
    ax2.set_ylabel("Количество точек", fontweight="bold")
    ax2.set_title("Распределение точек по кластерам", pad=15)

    ax2.set_xticks(unique_labels)
    ax2.set_xticklabels(
        [f"Кластер {i}" for i in unique_labels], rotation=45, ha="right"
    )
    ax2.grid(True, axis="y", alpha=0.4, linestyle="--")

    # Добавление значений на столбцы
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(counts) * 0.01,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#2C3E50",
        )

    # Статистика балансировки
    balance_stats = (
        f"Статистика баланса:\n"
        f"Средний размер: {np.mean(counts):.0f}\n"
        f"Стд. отклонение: {np.std(counts):.0f}\n"
        f"Коэф. вариации: {np.std(counts)/np.mean(counts):.2%}"
    )
    create_info_box(ax2, balance_stats, "top_right", "#F8F9FA", "#95A5A6")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Сохранено 2D: {save_path}")


def plot_dataset_3d(
    data: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    metadata: dict[str, Any],
    save_path: Path,
) -> None:
    """3D визуализация для датасетов с D >= 3."""
    if data.shape[1] < 3:
        print(f"  Пропуск 3D: данные имеют {data.shape[1]} измерений")
        return

    configure_plot_style()

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("3D Визуализация кластеров", y=1.02)

    # 3D scatter plot
    ax1 = fig.add_subplot(221, projection="3d")
    ax1.scatter(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        c=labels,
        cmap="tab20c",
        alpha=0.6,
        s=15,
        edgecolors="white",
        linewidth=0.3,
        depthshade=True,
    )

    ax1.scatter(
        centroids[:, 0],
        centroids[:, 1],
        centroids[:, 2],
        c="#E74C3C",
        marker="*",
        s=300,
        edgecolors="black",
        linewidth=2,
        label="Центроиды",
        zorder=10,
    )

    ax1.set_xlabel("Признак 1", fontweight="bold", labelpad=10)
    ax1.set_ylabel("Признак 2", fontweight="bold", labelpad=10)
    ax1.set_zlabel("Признак 3", fontweight="bold", labelpad=10)
    ax1.set_title("Объемное представление данных", pad=15)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.4)
    ax1.view_init(elev=25, azim=45)

    # 2D проекции
    proj_config = [
        (222, 0, 1, "XY Проекция"),
        (223, 0, 2, "XZ Проекция"),
        (224, 1, 2, "YZ Проекция"),
    ]

    for pos, x_idx, y_idx, title in proj_config:
        ax = fig.add_subplot(pos)
        ax.scatter(
            data[:, x_idx],
            data[:, y_idx],
            c=labels,
            cmap="tab20c",
            alpha=0.5,
            s=10,
            edgecolors="white",
            linewidth=0.2,
        )
        ax.scatter(
            centroids[:, x_idx],
            centroids[:, y_idx],
            c="#E74C3C",
            marker="*",
            s=200,
            edgecolors="black",
            linewidth=1.5,
            zorder=10,
        )

        ax.set_xlabel(f"Признак {x_idx+1}", fontweight="bold")
        ax.set_ylabel(f"Признак {y_idx+1}", fontweight="bold")
        ax.set_title(title, pad=12)
        ax.grid(True, alpha=0.4, linestyle="--")

    # Информация
    info_text = (
        f"Информация о датасете:\n"
        f"Всего точек: {len(data):,}\n"
        f"Размерность: {data.shape[1]}\n"
        f"Кластеров: {len(centroids)}\n"
        f"Показано: первые 3 измерения"
    )
    plt.figtext(
        0.02,
        0.02,
        info_text,
        fontsize=9,
        fontweight="bold",
        bbox=dict(
            boxstyle="round",
            facecolor="#FFF3CD",
            alpha=0.9,
            edgecolor="#F39C12",
            linewidth=2,
        ),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Сохранено 3D: {save_path}")


def plot_high_dim_stats(
    data: np.ndarray,
    labels: np.ndarray,
    metadata: dict[str, Any],
    save_path: Path,
) -> None:
    """Статистическая визуализация для многомерных датасетов."""
    configure_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Статистический анализ многомерного датасета", y=1.02)

    # 1. Статистика признаков
    ax1 = axes[0, 0]
    n_features = min(10, data.shape[1])
    feature_means = np.mean(data[:, :n_features], axis=0)
    feature_stds = np.std(data[:, :n_features], axis=0)

    x = np.arange(n_features)
    ax1.bar(
        x,
        feature_means,
        yerr=feature_stds,
        capsize=5,
        alpha=0.8,
        color="#3498DB",
        edgecolor="#2C3E50",
        linewidth=1.5,
    )

    ax1.set_xlabel("Индекс признака", fontweight="bold")
    ax1.set_ylabel("Среднее ± Стандартное отклонение", fontweight="bold")
    ax1.set_title(
        f"Распределение значений первых {n_features} признаков", pad=15
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"Признак {i+1}" for i in range(n_features)], rotation=45, ha="right"
    )
    ax1.grid(True, alpha=0.4, linestyle="--", axis="y")

    # 2. Распределение норм
    ax2 = axes[0, 1]
    norms = np.linalg.norm(data, axis=1)
    ax2.hist(
        norms,
        bins=50,
        alpha=0.8,
        color="#27AE60",
        edgecolor="#145A32",
        linewidth=1.5,
    )

    ax2.set_xlabel("Евклидова норма точки", fontweight="bold")
    ax2.set_ylabel("Частота", fontweight="bold")
    ax2.set_title("Распределение норм точек данных", pad=15)
    ax2.grid(True, alpha=0.4, linestyle="--")

    # Статистика норм
    norm_stats = (
        f"Статистика норм:\n"
        f"Среднее: {np.mean(norms):.2f}\n"
        f"Медиана: {np.median(norms):.2f}\n"
        f"Стд. отклонение: {np.std(norms):.2f}"
    )
    create_info_box(ax2, norm_stats, "top_right", "#E8F6F3", "#27AE60")

    # 3. PCA анализ
    ax3 = axes[1, 0]
    if data.shape[1] > 5:
        from sklearn.decomposition import PCA

        pca = PCA().fit(data)
        cumsum = np.cumsum(pca.explained_variance_ratio_)

        ax3.plot(
            range(1, len(cumsum) + 1),
            cumsum,
            "b-o",
            linewidth=2.5,
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=2,
            label="Накопленная дисперсия",
        )
        ax3.axhline(
            y=0.95,
            color="#E74C3C",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="95% дисперсии",
        )

        ax3.set_xlabel("Количество главных компонент", fontweight="bold")
        ax3.set_ylabel("Накопленная объясненная дисперсия", fontweight="bold")
        ax3.set_title("Анализ главных компонент (PCA)", pad=15)
        ax3.grid(True, alpha=0.4, linestyle="--")
        ax3.legend(fontsize=10)

        n_95 = np.argmax(cumsum >= 0.95) + 1
        ax3.axvline(x=n_95, color="#2ECC71", linestyle=":", linewidth=2, alpha=0.7)

        pca_info = (
            f"Результаты PCA:\n"
            f"Всего компонент: {data.shape[1]}\n"
            f"Для 95% дисперсии: {n_95}\n"
            f"Сжатие: {((data.shape[1] - n_95)/data.shape[1]*100):.1f}%"
        )
        create_info_box(ax3, pca_info, "bottom_left", "#EAF2F8", "#3498DB")
    else:
        ax3.text(
            0.5,
            0.5,
            f"Размерность D={data.shape[1]} слишком мала для PCA\n(требуется D > 5)",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=11,
        )
        ax3.set_title("PCA анализ не требуется")
        ax3.axis("off")

    # 4. Статистика кластеров
    ax4 = axes[1, 1]
    unique_labels, counts = np.unique(labels, return_counts=True)
    colors = plt.cm.tab20c(np.linspace(0, 1, len(unique_labels)))

    if len(unique_labels) <= 12:
        ax4.pie(
            counts,
            labels=[f"Кластер {i}" for i in unique_labels],
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 9, "fontweight": "bold"},
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        ax4.set_title("Процентное распределение по кластерам", pad=15)
    else:
        ax4.bar(
            unique_labels,
            counts,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )
        ax4.set_xlabel("ID кластера", fontweight="bold")
        ax4.set_ylabel("Количество точек", fontweight="bold")
        ax4.set_title("Распределение точек по кластерам", pad=15)
        ax4.grid(True, alpha=0.4, linestyle="--", axis="y")

        step = max(1, len(unique_labels) // 10)
        ax4.set_xticks(unique_labels[::step])
        ax4.set_xticklabels([f"{i}" for i in unique_labels[::step]])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Сохранено статистика: {save_path}")


def process_single_dataset(dataset_file: Path, output_dir: Path) -> bool:
    """Обработка одного датасета."""
    try:
        data, labels, centroids, metadata = load_dataset(dataset_file)

        print(
            f"   Загружено: {len(data):,} точек, {data.shape[1]}D, "
            f"{len(np.unique(labels))} кластеров"
        )

        # Создание структуры директорий
        rel_path = dataset_file.relative_to("datasets")
        output_subdir = output_dir / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)

        dataset_name = dataset_file.stem

        # Генерация визуализаций
        viz_2d_path = output_subdir / f"{dataset_name}_2d.png"
        plot_dataset_2d(data, labels, centroids, metadata, viz_2d_path)

        if data.shape[1] >= 3:
            viz_3d_path = output_subdir / f"{dataset_name}_3d.png"
            plot_dataset_3d(data, labels, centroids, metadata, viz_3d_path)

        if data.shape[1] >= 5:
            stats_path = output_subdir / f"{dataset_name}_stats.png"
            plot_high_dim_stats(data, labels, metadata, stats_path)

        # Сохранение информации
        save_dataset_info(output_subdir, dataset_name, dataset_file, data, labels, metadata)

        return True

    except Exception as e:  # noqa: BLE001
        print(f"   Ошибка при обработке: {e}")
        return False


def save_dataset_info(
    output_dir: Path,
    dataset_name: str,
    dataset_file: Path,
    data: np.ndarray,
    labels: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Сохранение информации о датасете."""
    info_path = output_dir / f"{dataset_name}_info.txt"
    cluster_sizes = dict(zip(*np.unique(labels, return_counts=True)))

    with info_path.open("w", encoding="utf-8") as f:
        f.write(f"Датасет: {dataset_name}\n")
        f.write(f"Файл: {dataset_file}\n")
        f.write(f"Размер данных: {data.shape}\n")
        f.write(f"Количество кластеров: {len(np.unique(labels))}\n")
        f.write(f"Размеры кластеров: {cluster_sizes}\n")
        f.write("\nМетаданные:\n")
        f.write(json.dumps(metadata, indent=2, ensure_ascii=False))
        f.write(
            f"\n\nСгенерировано: "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )


def visualize_all_datasets(paths: DatasetPaths | None = None) -> None:
    """Основная функция для визуализации всех датасетов."""
    if paths is None:
        paths = DatasetPaths()

    datasets_dir = paths.datasets_dir
    output_dir = paths.output_dir
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("ВИЗУАЛИЗАЦИЯ ДАТАСЕТОВ")
    print("=" * 60)
    print()

    dataset_files = list(datasets_dir.glob("**/*.txt"))
    print(f"Найдено файлов датасетов: {len(dataset_files)}")
    print("-" * 60)

    successful = 0
    for i, dataset_file in enumerate(dataset_files, 1):
        print(f"\n[{i}/{len(dataset_files)}] Обработка: {dataset_file.name}")

        if process_single_dataset(dataset_file, output_dir):
            successful += 1

    # Создание HTML отчета
    create_html_summary(output_dir, dataset_files, successful)

    print("\n" + "=" * 60)
    print("ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА")
    print(f"Успешно обработано: {successful}/{len(dataset_files)}")
    print(f"Выходная директория: {output_dir.absolute()}")
    print("=" * 60)


def create_html_summary(
    output_dir: Path, dataset_files: Sequence[Path], successful_count: int
) -> None:
    """Создание HTML отчета со всеми визуализациями."""
    html_path = output_dir / "summary.html"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Группировка датасетов
    datasets_by_dir: dict[str, list[str]] = defaultdict(list)
    for dataset_file in dataset_files:
        try:
            rel_path = dataset_file.relative_to("datasets")
            datasets_by_dir[str(rel_path.parent)].append(dataset_file.stem)
        except ValueError:
            datasets_by_dir["."].append(dataset_file.stem)

    html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Отчет по визуализации датасетов K-means</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            margin: 0;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #3498db;
        }}
        
        .header h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #3498db;
        }}
        
        .stat-card h3 {{
            color: #2c3e50;
            margin-bottom: 5px;
            font-size: 1.5em;
        }}
        
        .directory-section {{
            margin-bottom: 30px;
            padding: 20px;
            background: #fafafa;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }}
        
        .directory-section h2 {{
            color: #34495e;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .dataset-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #dee2e6;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        
        .dataset-card h3 {{
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.2em;
        }}
        
        .images-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        
        .image-container {{
            text-align: center;
        }}
        
        .image-container h4 {{
            color: #495057;
            margin-bottom: 10px;
            font-size: 0.9em;
        }}
        
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            border: 1px solid #ddd;
            transition: transform 0.2s;
        }}
        
        .image-container img:hover {{
            transform: scale(1.02);
        }}
        
        .filter-section {{
            margin: 20px 0;
            padding: 15px;
            background: #e8f4fc;
            border-radius: 8px;
        }}
        
        .filter-input {{
            width: 100%;
            padding: 10px;
            border: 2px solid #3498db;
            border-radius: 5px;
            font-size: 14px;
        }}
        
        @media (max-width: 768px) {{
            .images-grid {{
                grid-template-columns: 1fr;
            }}
            
            .stats-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Отчет по визуализации датасетов K-means</h1>
            <p>Сравнительный анализ синтетических данных для бенчмаркинга</p>
        </div>
        
        <div class="filter-section">
            <input type="text" class="filter-input" 
                   placeholder="Поиск датасетов по названию..." 
                   onkeyup="filterDatasets()">
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>{len(dataset_files)}</h3>
                <p>Всего датасетов</p>
            </div>
            <div class="stat-card">
                <h3>{successful_count}</h3>
                <p>Успешно обработано</p>
            </div>
            <div class="stat-card">
                <h3>{len(datasets_by_dir)}</h3>
                <p>Директорий</p>
            </div>
            <div class="stat-card">
                <h3>{current_time}</h3>
                <p>Время генерации</p>
            </div>
        </div>
    """

    # Добавление датасетов по директориям
    for dir_name in sorted(datasets_by_dir.keys()):
        display_name = dir_name if dir_name != "." else "Корневая директория"
        html_content += f"""
        <div class="directory-section">
            <h2>Директория: {display_name}</h2>
        """

        for dataset_name in sorted(datasets_by_dir[dir_name]):
            html_content += f"""
            <div class="dataset-card" data-name="{dataset_name.lower()}">
                <h3>{dataset_name}</h3>
                <div class="images-grid">
            """

            viz_dir = output_dir / dir_name
            viz_types = [
                ("2d", "2D Визуализация"),
                ("3d", "3D Визуализация"),
                ("stats", "Статистика"),
            ]

            for viz_type, viz_title in viz_types:
                img_path = viz_dir / f"{dataset_name}_{viz_type}.png"
                if img_path.exists():
                    rel_path = img_path.relative_to(output_dir)
                    html_content += f"""
                    <div class="image-container">
                        <h4>{viz_title}</h4>
                        <img src="{rel_path}" alt="{viz_title}" 
                             onclick="zoomImage(this)">
                    </div>
                    """

            html_content += """
                </div>
            </div>
            """

        html_content += "</div>"

    html_content += """
    </div>
    
    <script>
        function filterDatasets() {
            const input = document.querySelector('.filter-input');
            const filter = input.value.toLowerCase();
            const datasets = document.querySelectorAll('.dataset-card');
            
            datasets.forEach(dataset => {
                const name = dataset.getAttribute('data-name');
                if (name.includes(filter)) {
                    dataset.style.display = 'block';
                } else {
                    dataset.style.display = 'none';
                }
            });
        }
        
        function zoomImage(img) {
            if (img.classList.contains('zoomed')) {
                img.classList.remove('zoomed');
                img.style.maxWidth = '100%';
                img.style.cursor = 'zoom-in';
            } else {
                img.classList.add('zoomed');
                img.style.maxWidth = '90vw';
                img.style.cursor = 'zoom-out';
            }
        }
        
        document.querySelectorAll('.image-container img').forEach(img => {
            img.style.cursor = 'zoom-in';
        });
    </script>
</body>
</html>
    """

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML отчет создан: {html_path}")


def _cli() -> None:
    """CLI-обёртка для визуализации датасетов."""
    parser = argparse.ArgumentParser(
        description=(
            "Визуализация синтетических датасетов K-means "
            "и генерация HTML-отчёта."
        )
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("datasets"),
        help="Директория с датасетами (по умолчанию: ./datasets)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualizations"),
        help="Директория для сохранения визуализаций (по умолчанию: ./visualizations)",
    )

    args = parser.parse_args()

    paths = DatasetPaths(datasets_dir=args.datasets_dir, output_dir=args.output_dir)
    visualize_all_datasets(paths)


if __name__ == "__main__":
    _cli()