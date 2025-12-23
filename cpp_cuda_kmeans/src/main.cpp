#include <iostream>
#include <string>

#include "experiments/config.h"
#include "experiments/suite.h"
#include "utils/logging.h"

namespace {

std::string get_arg(int argc, char **argv, const std::string &name, const std::string &default_value) {
    for (int i = 1; i < argc - 1; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return default_value;
}

bool has_flag(int argc, char **argv, const std::string &flag) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == flag) {
            return true;
        }
    }
    return false;
}

void print_help() {
    std::cout << "Usage: kmeans_benchmark --experiment <id> [--output <path>]\n";
    std::cout << "Experiments: exp2_scaling_n, exp3_scaling_d, exp4_scaling_k, all\n";
}

}  // namespace

int main(int argc, char **argv) {
    if (has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
        print_help();
        return 0;
    }

    std::string experiment_value = get_arg(argc, argv, "--experiment", "all");
    std::string output_path = get_arg(argc, argv, "--output", "kmeans_timing_results.json");

    try {
        auto experiment_id = kmeans::experiments::parse_experiment_id(experiment_value);
        kmeans::experiments::ExperimentSuite suite(kmeans::utils::Logger("suite"));
        suite.run(experiment_id, output_path);
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
