#pragma once

#include <cstddef>
#include <span>
#include <optional>
#include <string_view>
#include <vector>

namespace cpp_knn {

struct Matrix {
    std::size_t rows{};
    std::size_t columns{};
    std::vector<double> values;

    [[nodiscard]] std::span<const double> row(std::size_t index) const;
};

enum class SelectionMethod {
    heap,
    nth_element,
};

enum class WeightMode { uniform, distance };
[[nodiscard]] WeightMode parse_weight_mode(std::string_view value);
[[nodiscard]] std::string_view weight_mode_name(WeightMode value);

[[nodiscard]] SelectionMethod parse_selection_method(std::string_view value);
[[nodiscard]] std::string_view selection_method_name(SelectionMethod value);

struct PredictionResult {
    std::vector<int> labels;
    std::vector<std::size_t> positive_vote_counts;
    std::vector<double> scores;
    double distance_seconds{};
    double selection_vote_seconds{};
};

class ExactKnnClassifier {
public:
    ExactKnnClassifier(Matrix training, std::vector<int> labels, std::size_t k,
                       WeightMode weights = WeightMode::uniform,
                       std::optional<double> threshold = std::nullopt);

    [[nodiscard]] PredictionResult predict_reference(const Matrix& queries) const;
    [[nodiscard]] PredictionResult predict_optimized(
        const Matrix& queries,
        std::size_t batch_size,
        SelectionMethod selection_method
    ) const;

    [[nodiscard]] std::size_t training_rows() const noexcept;
    [[nodiscard]] std::size_t feature_count() const noexcept;
    [[nodiscard]] std::size_t k() const noexcept;
    [[nodiscard]] std::size_t auxiliary_bytes() const noexcept;

private:
    struct Neighbour {
        double squared_distance;
        std::size_t index;
    };

    struct BetterNeighbour {
        [[nodiscard]] bool operator()(const Neighbour& left,
                                      const Neighbour& right) const noexcept;
    };

    [[nodiscard]] std::vector<Neighbour> reference_neighbours(
        std::span<const double> query
    ) const;
    [[nodiscard]] std::vector<Neighbour> nth_neighbours(
        const std::vector<double>& distances,
        std::size_t query_index,
        std::size_t query_count,
        std::vector<Neighbour>& candidates
    ) const;
    void validate_queries(const Matrix& queries) const;
    void vote(const std::vector<Neighbour>& neighbours, PredictionResult& output,
              std::size_t row) const;

    Matrix training_;
    std::vector<int> labels_;
    std::size_t k_;
    WeightMode weights_;
    std::optional<double> threshold_;
};

}  // namespace cpp_knn
