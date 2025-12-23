#include "experiments/suite.h"

#include <fstream>
#include <thread>

#include "core/kmeans_cpu.h"
#include "core/kmeans_cpu_omp.h"
#include "core/kmeans_gpu.h"
#include "data/dataset.h"
#include "data/validation.h"
#include "experiments/runner.h"
#include "metrics/metrics.h"

namespace kmeans::experiments {

namespace {

void append_result(const std::string &path, const std::string &experiment, const std::string &implementation,
                   const kmeans::data::Dataset &dataset, const TimingSummary &timing, double speedup,
                   double efficiency) {
    std::ofstream out(path, std::ios::app);
    out << "{";
    out << "\"experiment\":\"" << experiment << "\",";
    out << "\"implementation\":\"" << implementation << "\",";
    out << "\"dataset\":{";
    out << "\"N\":" << dataset.n_samples << ",";
    out << "\"D\":" << dataset.n_features << ",";
    out << "\"K\":" << dataset.n_clusters << "},";
    out << "\"timing\":{";
    out << "\"T_fit_avg\":" << timing.total_ms << ",";
    out << "\"T_assign_total_avg\":" << timing.assign_ms << ",";
    out << "\"T_update_total_avg\":" << timing.update_ms << ",";
    out << "\"speedup\":" << speedup << ",";
    out << "\"efficiency\":" << efficiency;
    out << "}}\n";
}

void run_for_dataset(const std::string &experiment_name, const kmeans::data::Dataset &dataset, int repeats,
                     const std::string &output_path, kmeans::utils::Logger logger) {
    kmeans::core::KMeansCPU cpu(dataset.n_clusters, 100, 1e-4f, kmeans::utils::Logger("cpu"));
    auto cpu_result = run_single(cpu, dataset, repeats);
    cpu_result.implementation = "cpp_cpu_single";

    int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    kmeans::core::KMeansCPUOpenMP cpu_omp(dataset.n_clusters, 100, 1e-4f, kmeans::utils::Logger("cpu_openmp"),
                                          threads);
    auto omp_result = run_single(cpu_omp, dataset, repeats);
    omp_result.implementation = "cpp_cpu_openmp";

    kmeans::core::KMeansGPU gpu(dataset.n_clusters, 100, 1e-4f, kmeans::utils::Logger("gpu"));
    auto gpu_result = run_single(gpu, dataset, repeats);
    gpu_result.implementation = "cpp_gpu_cuda";

    double cpu_total = cpu_result.timing.total_ms;

    append_result(output_path, experiment_name, cpu_result.implementation, dataset, cpu_result.timing, 1.0,
                  kmeans::metrics::compute_efficiency(1.0, 1));

    double omp_speedup = kmeans::metrics::compute_speedup(cpu_total, omp_result.timing.total_ms);
    double omp_eff = kmeans::metrics::compute_efficiency(omp_speedup, threads);
    append_result(output_path, experiment_name, omp_result.implementation, dataset, omp_result.timing, omp_speedup,
                  omp_eff);

    double gpu_speedup = kmeans::metrics::compute_speedup(cpu_total, gpu_result.timing.total_ms);
    append_result(output_path, experiment_name, gpu_result.implementation, dataset, gpu_result.timing, gpu_speedup,
                  1.0);

    logger.info("Completed dataset " + dataset.name);
}

}  // namespace

ExperimentSuite::ExperimentSuite(kmeans::utils::Logger logger) : logger_(std::move(logger)) {}

void ExperimentSuite::run(ExperimentId id, const std::string &output_path) {
    if (id == ExperimentId::kAll) {
        run_scaling_n(output_path);
        run_scaling_d(output_path);
        run_scaling_k(output_path);
        return;
    }

    if (id == ExperimentId::kExp2ScalingN) {
        run_scaling_n(output_path);
        return;
    }
    if (id == ExperimentId::kExp3ScalingD) {
        run_scaling_d(output_path);
        return;
    }
    if (id == ExperimentId::kExp4ScalingK) {
        run_scaling_k(output_path);
    }
}

void ExperimentSuite::run_scaling_n(const std::string &output_path) {
    auto config = experiment_configs().at(ExperimentId::kExp2ScalingN);
    for (int value : config.values) {
        auto dataset = kmeans::data::Dataset::random_gaussian("scaling_n_" + std::to_string(value), value, 32, 8, 42);
        if (!kmeans::data::validate_dataset(dataset)) {
            logger_.error("Invalid dataset for scaling N");
            continue;
        }
        run_for_dataset("exp2_scaling_n", dataset, config.repeats, output_path, logger_);
    }
}

void ExperimentSuite::run_scaling_d(const std::string &output_path) {
    auto config = experiment_configs().at(ExperimentId::kExp3ScalingD);
    for (int value : config.values) {
        auto dataset = kmeans::data::Dataset::random_gaussian("scaling_d_" + std::to_string(value), 50000, value, 8, 42);
        if (!kmeans::data::validate_dataset(dataset)) {
            logger_.error("Invalid dataset for scaling D");
            continue;
        }
        run_for_dataset("exp3_scaling_d", dataset, config.repeats, output_path, logger_);
    }
}

void ExperimentSuite::run_scaling_k(const std::string &output_path) {
    auto config = experiment_configs().at(ExperimentId::kExp4ScalingK);
    for (int value : config.values) {
        auto dataset = kmeans::data::Dataset::random_gaussian("scaling_k_" + std::to_string(value), 50000, 32, value, 42);
        if (!kmeans::data::validate_dataset(dataset)) {
            logger_.error("Invalid dataset for scaling K");
            continue;
        }
        run_for_dataset("exp4_scaling_k", dataset, config.repeats, output_path, logger_);
    }
}

}  // namespace kmeans::experiments
