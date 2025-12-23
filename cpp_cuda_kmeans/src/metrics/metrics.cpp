#include "metrics/metrics.h"

namespace kmeans::metrics {

double compute_speedup(double baseline_ms, double optimized_ms) {
    if (optimized_ms <= 0.0) {
        return 0.0;
    }
    return baseline_ms / optimized_ms;
}

double compute_efficiency(double speedup, int parallelism) {
    if (parallelism <= 0) {
        return 0.0;
    }
    return speedup / static_cast<double>(parallelism);
}

double compute_throughput(double operations, double total_ms) {
    if (total_ms <= 0.0) {
        return 0.0;
    }
    return operations / (total_ms / 1000.0);
}

}  // namespace kmeans::metrics
