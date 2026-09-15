#pragma once

#include "cpp_knn/knn.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace cpp_knn {

[[nodiscard]] Matrix load_feature_csv(const std::filesystem::path& path);
[[nodiscard]] std::vector<int> load_label_csv(const std::filesystem::path& path);
[[nodiscard]] std::string prediction_hash(
    const std::vector<int>& labels,
    const std::vector<std::size_t>& positive_vote_counts
);

}  // namespace cpp_knn
