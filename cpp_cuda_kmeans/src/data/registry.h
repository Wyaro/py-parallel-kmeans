#pragma once

#include <string>
#include <unordered_map>

#include "data/dataset.h"

namespace kmeans::data {

class DatasetRegistry {
public:
    void add(const Dataset &dataset);
    const Dataset &get(const std::string &name) const;
    bool contains(const std::string &name) const;

private:
    std::unordered_map<std::string, Dataset> datasets_{};
};

}  // namespace kmeans::data
