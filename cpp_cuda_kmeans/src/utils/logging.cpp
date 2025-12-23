#include "utils/logging.h"

#include <iostream>

namespace kmeans::utils {

Logger::Logger(std::string name) : name_(std::move(name)) {}

void Logger::log(LogLevel level, const std::string &message) const {
    const char *level_str = "INFO";
    switch (level) {
        case LogLevel::kInfo:
            level_str = "INFO";
            break;
        case LogLevel::kWarning:
            level_str = "WARN";
            break;
        case LogLevel::kError:
            level_str = "ERROR";
            break;
    }
    std::cout << "[" << level_str << "] " << name_ << ": " << message << "\n";
}

void Logger::info(const std::string &message) const {
    log(LogLevel::kInfo, message);
}

void Logger::warn(const std::string &message) const {
    log(LogLevel::kWarning, message);
}

void Logger::error(const std::string &message) const {
    log(LogLevel::kError, message);
}

}  // namespace kmeans::utils
