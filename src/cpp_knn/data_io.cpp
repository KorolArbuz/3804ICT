#include "cpp_knn/data_io.hpp"

#include <cerrno>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cpp_knn {
namespace {

[[nodiscard]] double parse_double(std::string_view token,
                                  const std::filesystem::path& path,
                                  std::size_t line_number) {
    if (token.empty()) {
        throw std::runtime_error("empty value in " + path.string() + ":" +
                                 std::to_string(line_number));
    }
    const std::string owned(token);
    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(owned.c_str(), &end);
    if (errno == ERANGE || end != owned.c_str() + owned.size() ||
        !std::isfinite(value)) {
        throw std::runtime_error("invalid finite double in " + path.string() +
                                 ":" + std::to_string(line_number));
    }
    return value;
}

void remove_carriage_return(std::string& line) {
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
}

void fnv_update(std::uint64_t& hash, std::string_view value) {
    constexpr std::uint64_t prime = 1099511628211ULL;
    for (const unsigned char byte : value) {
        hash ^= byte;
        hash *= prime;
    }
}

}  // namespace

Matrix load_feature_csv(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open feature CSV: " + path.string());
    }
    Matrix matrix;
    std::string line;
    while (std::getline(input, line)) {
        ++matrix.rows;
        remove_carriage_return(line);
        if (line.empty()) {
            throw std::runtime_error("blank row in feature CSV: " + path.string());
        }
        std::size_t row_columns = 0;
        std::size_t begin = 0;
        while (true) {
            const std::size_t comma = line.find(',', begin);
            const std::size_t end = comma == std::string::npos ? line.size() : comma;
            matrix.values.push_back(parse_double(
                std::string_view(line).substr(begin, end - begin), path, matrix.rows
            ));
            ++row_columns;
            if (comma == std::string::npos) {
                break;
            }
            begin = comma + 1;
        }
        if (matrix.columns == 0) {
            matrix.columns = row_columns;
        } else if (row_columns != matrix.columns) {
            throw std::runtime_error("inconsistent feature count in " +
                                     path.string() + ":" +
                                     std::to_string(matrix.rows));
        }
    }
    if (!input.eof()) {
        throw std::runtime_error("failed while reading feature CSV: " + path.string());
    }
    if (matrix.rows == 0 || matrix.columns == 0) {
        throw std::runtime_error("feature CSV is empty: " + path.string());
    }
    return matrix;
}

std::vector<int> load_label_csv(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open label CSV: " + path.string());
    }
    std::vector<int> labels;
    std::string line;
    while (std::getline(input, line)) {
        remove_carriage_return(line);
        if (line != "0" && line != "1") {
            throw std::runtime_error("labels must be one binary value per line in " +
                                     path.string());
        }
        labels.push_back(line == "1" ? 1 : 0);
    }
    if (!input.eof()) {
        throw std::runtime_error("failed while reading label CSV: " + path.string());
    }
    if (labels.empty()) {
        throw std::runtime_error("label CSV is empty: " + path.string());
    }
    return labels;
}

std::string prediction_hash(
    const std::vector<int>& labels,
    const std::vector<std::size_t>& positive_vote_counts) {
    if (labels.size() != positive_vote_counts.size()) {
        throw std::invalid_argument("hash inputs have different lengths");
    }
    std::uint64_t hash = 14695981039346656037ULL;
    for (std::size_t index = 0; index < labels.size(); ++index) {
        fnv_update(hash, std::to_string(index));
        fnv_update(hash, ",");
        fnv_update(hash, std::to_string(labels[index]));
        fnv_update(hash, ",");
        fnv_update(hash, std::to_string(positive_vote_counts[index]));
        fnv_update(hash, "\n");
    }
    std::ostringstream text;
    text << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << hash;
    return text.str();
}

}  // namespace cpp_knn
