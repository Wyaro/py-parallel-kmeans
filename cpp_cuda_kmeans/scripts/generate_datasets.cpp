#include <fstream>
#include <iostream>

#include "data/dataset.h"

int main() {
    auto dataset = kmeans::data::Dataset::random_gaussian("demo", 1000, 16, 4, 42);

    std::ofstream out("dataset_demo.csv");
    for (int i = 0; i < dataset.n_samples; ++i) {
        for (int d = 0; d < dataset.n_features; ++d) {
            out << dataset.values[static_cast<size_t>(i) * dataset.n_features + d];
            if (d + 1 < dataset.n_features) {
                out << ',';
            }
        }
        out << '\n';
    }

    std::cout << "Generated dataset_demo.csv with " << dataset.n_samples << " samples\n";
    return 0;
}
