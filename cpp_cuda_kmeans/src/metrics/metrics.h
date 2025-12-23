#pragma once

namespace kmeans::metrics {

double compute_speedup(double baseline_ms, double optimized_ms);

double compute_efficiency(double speedup, int parallelism);

double compute_throughput(double operations, double total_ms);

}  // namespace kmeans::metrics
