#include "metrics/timers.h"

namespace kmeans::metrics {

void Timer::start() {
    start_time_ = std::chrono::steady_clock::now();
}

double Timer::stop() {
    const auto end_time = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = end_time - start_time_;
    return elapsed.count();
}

}  // namespace kmeans::metrics
