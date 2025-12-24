"""
Расчёт производственных метрик на основе текстового summary-файла.

Скрипт парсит `analysis_summary.txt`, сформированный анализом таймингов,
и вычисляет ряд метрик (ускорение, эффективность, пропускная способность)
для различных реализаций K-means (CPU / GPU / multiprocessing).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argparse


def parse_summary_file(filepath: str | Path) -> List[Dict]:
    """Парсит файл analysis_summary.txt и извлекает данные о результатах экспериментов."""
    results: list[dict] = []

    filepath = Path(filepath)
    with filepath.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Ищем строку с экспериментом
        if line.startswith('Эксперимент:'):
            parts = line.split(',')
            if len(parts) >= 2:
                experiment = parts[0].replace('Эксперимент:', '').strip()
                implementation = parts[1].replace('реализация:', '').strip()
                
                # Читаем следующие строки
                i += 1
                dataset_info = {}
                n_iters = None
                T_assign = None
                T_update = None
                T_iter = None
                T_total = None
                
                while i < len(lines) and not lines[i].strip().startswith('Эксперимент:'):
                    line = lines[i].strip()
                    
                    if line.startswith('Датасет:'):
                        # Извлекаем N, D, K
                        n_match = re.search(r'N=(\d+)', line)
                        d_match = re.search(r'D=(\d+)', line)
                        k_match = re.search(r'K=(\d+)', line)
                        if n_match:
                            dataset_info['N'] = int(n_match.group(1))
                        if d_match:
                            dataset_info['D'] = int(d_match.group(1))
                        if k_match:
                            dataset_info['K'] = int(k_match.group(1))
                    
                    elif 'Среднее количество итераций:' in line:
                        n_iters_match = re.search(r'([\d.]+)', line)
                        if n_iters_match:
                            n_iters = float(n_iters_match.group(1))
                    
                    elif 'Tназначения' in line and 'ср=' in line:
                        match = re.search(r'ср=([\d.]+)', line)
                        if match:
                            T_assign = float(match.group(1))
                    
                    elif 'Tобновления' in line and 'ср=' in line:
                        match = re.search(r'ср=([\d.]+)', line)
                        if match:
                            T_update = float(match.group(1))
                    
                    elif 'Tитерации' in line and 'ср=' in line:
                        match = re.search(r'ср=([\d.]+)', line)
                        if match:
                            T_iter = float(match.group(1))
                    
                    elif 'Tобщ' in line and 'ср=' in line:
                        match = re.search(r'ср=([\d.]+)', line)
                        if match:
                            T_total = float(match.group(1))
                    
                    i += 1
                
                # Если собрали все данные, добавляем результат
                if dataset_info and n_iters is not None and T_total is not None:
                    # Извлекаем количество потоков из названия реализации
                    threads = None
                    if 'mp_' in implementation:
                        threads_match = re.search(r'mp_(\d+)', implementation)
                        if threads_match:
                            threads = int(threads_match.group(1))
                    
                    results.append({
                        'experiment': experiment,
                        'implementation': implementation,
                        'N': dataset_info.get('N', 0),
                        'D': dataset_info.get('D', 0),
                        'K': dataset_info.get('K', 0),
                        'n_iters': n_iters,
                        'T_assign': T_assign or 0.0,
                        'T_update': T_update or 0.0,
                        'T_iter': T_iter or 0.0,
                        'T_total': T_total,  # в мс
                        'threads': threads,
                    })
                continue
        
        i += 1
    
    return results


def calculate_gpu_blocks(N: int, threads_per_block: int = 256) -> int:
    """Вычисляет количество блоков GPU на основе N."""
    return (N + threads_per_block - 1) // threads_per_block


def calculate_metrics(results: List[Dict]) -> List[Dict]:
    """Вычисляет метрики производительности для каждого результата."""
    # Группируем результаты по эксперименту и датасету
    grouped: Dict[Tuple[str, int, int, int], List[Dict]] = {}
    
    for result in results:
        key = (result['experiment'], result['N'], result['D'], result['K'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)
    
    # Вычисляем метрики для каждого результата
    metrics_results = []
    
    for key, group_results in grouped.items():
        experiment, N, D, K = key
        
        # Находим baseline (однопоточную реализацию)
        baseline = None
        for r in group_results:
            if r['implementation'] == 'python_cpu_numpy':
                baseline = r
                break
        
        if baseline is None:
            # Если нет baseline, пропускаем расчет ускорения и эффективности
            baseline = None
        
        for result in group_results:
            impl = result['implementation']
            
            # Пропускаем baseline сам по себе
            if impl == 'python_cpu_numpy':
                continue
            
            metrics = {
                'experiment': experiment,
                'implementation': impl,
                'N': N,
                'D': D,
                'K': K,
            }
            
            T_total_sec = result['T_total'] / 1000.0  # конвертируем мс в секунды
            
            # 1. Ускорение (Speedup): S = T_опр / T_мпр
            if baseline is not None:
                T_baseline_sec = baseline['T_total'] / 1000.0
                if T_total_sec > 0:
                    speedup = T_baseline_sec / T_total_sec
                    metrics['speedup'] = speedup
                else:
                    metrics['speedup'] = None
            else:
                metrics['speedup'] = None
            
            # 2. Параллельная эффективность: E = S / p
            p = None
            
            # Для CPU multiprocessing
            if result['threads'] is not None:
                p = result['threads']
                metrics['parallelism_factor'] = p
                metrics['parallelism_type'] = 'CPU_threads'
            
            # Для GPU
            elif impl.startswith('python_gpu_cupy'):
                threads_per_block = 256
                blocks = calculate_gpu_blocks(N, threads_per_block)
                p = blocks
                metrics['parallelism_factor'] = p
                metrics['parallelism_type'] = 'GPU_blocks'
                metrics['gpu_blocks'] = blocks
                metrics['gpu_threads_per_block'] = threads_per_block
            
            if metrics.get('speedup') is not None and p is not None and p > 0:
                efficiency = metrics['speedup'] / p
                metrics['efficiency'] = efficiency
            else:
                metrics['efficiency'] = None
            
            # 3. Пропускная способность: ПС = (N × K × D × N_итер) / T_общ
            n_iters = result['n_iters']
            if T_total_sec > 0:
                throughput = (N * K * D * n_iters) / T_total_sec
                metrics['throughput'] = throughput
            else:
                metrics['throughput'] = None
            
            # 4. Время передачи данных (GPU) - данных нет в summary
            if impl.startswith('python_gpu_cupy'):
                metrics['T_transfer'] = None  # Нет данных в summary
                metrics['T_transfer_ratio'] = None  # Нет данных в summary
            else:
                metrics['T_transfer'] = None
                metrics['T_transfer_ratio'] = None
            
            metrics['T_total_ms'] = result['T_total']
            metrics['T_total_sec'] = T_total_sec
            metrics['n_iters'] = n_iters
            
            metrics_results.append(metrics)
    
    return metrics_results


def format_metrics_output(metrics_results: List[Dict]) -> str:
    """Форматирует результаты метрик для вывода."""
    output_lines = []
    
    # Группируем по эксперименту и датасету
    grouped: Dict[Tuple[str, int, int, int], List[Dict]] = {}
    for m in metrics_results:
        key = (m['experiment'], m['N'], m['D'], m['K'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(m)
    
    for key, group in sorted(grouped.items()):
        experiment, N, D, K = key
        output_lines.append(f"\n{'='*80}")
        output_lines.append(f"Эксперимент: {experiment}, Датасет: N={N}, D={D}, K={K}")
        output_lines.append(f"{'='*80}")
        
        for m in sorted(group, key=lambda x: x['implementation']):
            impl = m['implementation']
            output_lines.append(f"\nРеализация: {impl}")
            
            if m.get('parallelism_type'):
                p_type = m['parallelism_type']
                p_factor = m.get('parallelism_factor', 'N/A')
                if p_type == 'GPU_blocks':
                    output_lines.append(f"  Параллелизм: {p_factor} блоков GPU (threads_per_block={m.get('gpu_threads_per_block', 256)})")
                else:
                    output_lines.append(f"  Параллелизм: {p_factor} потоков CPU")
            
            if m.get('speedup') is not None:
                output_lines.append(f"  Ускорение (Speedup): {m['speedup']:.4f}")
            else:
                output_lines.append(f"  Ускорение (Speedup): N/A (нет baseline)")
            
            if m.get('efficiency') is not None:
                output_lines.append(f"  Параллельная эффективность: {m['efficiency']:.6f}")
            else:
                output_lines.append(f"  Параллельная эффективность: N/A")
            
            if m.get('throughput') is not None:
                throughput = m['throughput']
                if throughput >= 1e9:
                    output_lines.append(f"  Пропускная способность: {throughput/1e9:.4f} x 10^9 оп/с")
                elif throughput >= 1e6:
                    output_lines.append(f"  Пропускная способность: {throughput/1e6:.4f} x 10^6 оп/с")
                elif throughput >= 1e3:
                    output_lines.append(f"  Пропускная способность: {throughput/1e3:.4f} x 10^3 оп/с")
                else:
                    output_lines.append(f"  Пропускная способность: {throughput:.4f} оп/с")
            else:
                output_lines.append(f"  Пропускная способность: N/A")
            
            if m.get('T_transfer') is not None:
                output_lines.append(f"  Время передачи данных (GPU): {m['T_transfer']:.6f} с")
            else:
                output_lines.append(f"  Время передачи данных (GPU): N/A (нет данных в summary)")
            
            if m.get('T_transfer_ratio') is not None:
                output_lines.append(f"  Доля времени на передачу данных: {m['T_transfer_ratio']:.2f}%")
            else:
                output_lines.append(f"  Доля времени на передачу данных: N/A (нет данных в summary)")
    
    return '\n'.join(output_lines)


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(
        description=(
            "Расчёт производственных метрик на основе файла analysis_summary.txt."
        )
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        default=Path("analysis_summary.txt"),
        help="Входной summary-файл (по умолчанию: ./analysis_summary.txt)",
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        type=Path,
        default=Path("metrics_summary.txt"),
        help="Файл для записи результатов (по умолчанию: ./metrics_summary.txt)",
    )

    args = parser.parse_args()

    print(f"Чтение данных из {args.input_file}...")
    results = parse_summary_file(args.input_file)
    print(f"Найдено {len(results)} результатов экспериментов")
    
    print("Вычисление метрик...")
    metrics_results = calculate_metrics(results)
    print(f"Вычислено метрик для {len(metrics_results)} реализаций")
    
    print("Форматирование результатов...")
    output = format_metrics_output(metrics_results)
    
    # Выводим без специальных символов для Windows консоли
    try:
        print(output)
    except UnicodeEncodeError:
        # Если не получается вывести, просто сохраняем в файл
        print("Результаты сохранены в файл (проблема с кодировкой консоли)")
    
    print(f"\nСохранение результатов в {args.output_file}...")
    with args.output_file.open('w', encoding='utf-8') as f:
        f.write(output)
    
    print("Готово!")


if __name__ == '__main__':
    main()

