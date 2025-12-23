#pragma once

#include <chrono>

namespace kmeans::metrics {

class Timer {
public:
    void start();
    double stop();

private:
    std::chrono::steady_clock::time_point start_time_{};
};

}  // namespace kmeans::metrics
