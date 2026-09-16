#include "cpp_knn/knn.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cpp_knn {
namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] double elapsed_seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void validate_matrix_storage(const Matrix& matrix, std::string_view name,
                             bool allow_empty) {
    if (matrix.columns == 0) {
        throw std::invalid_argument(std::string(name) +
                                    " must have at least one feature");
    }
    if (!allow_empty && matrix.rows == 0) {
        throw std::invalid_argument(std::string(name) + " must not be empty");
    }
    if (matrix.rows > std::numeric_limits<std::size_t>::max() / matrix.columns ||
        matrix.values.size() != matrix.rows * matrix.columns) {
        throw std::invalid_argument(std::string(name) +
                                    " dimensions do not match its values");
    }
    for (double value : matrix.values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(std::string(name) +
                                        " must contain only finite values");
        }
    }
}

}  // namespace

WeightMode parse_weight_mode(std::string_view value) {
    if (value == "uniform") return WeightMode::uniform;
    if (value == "distance") return WeightMode::distance;
    throw std::invalid_argument("weights must be uniform or distance");
}

std::string_view weight_mode_name(WeightMode value) {
    return value == WeightMode::uniform ? "uniform" : "distance";
}

std::span<const double> Matrix::row(std::size_t index) const {
    if (index >= rows) {
        throw std::out_of_range("matrix row index is out of range");
    }
    return std::span<const double>(values.data() + index * columns, columns);
}

SelectionMethod parse_selection_method(std::string_view value) {
    if (value == "heap") {
        return SelectionMethod::heap;
    }
    if (value == "nth") {
        return SelectionMethod::nth_element;
    }
    throw std::invalid_argument("selection must be 'heap' or 'nth'");
}

std::string_view selection_method_name(SelectionMethod value) {
    return value == SelectionMethod::heap ? "heap" : "nth_element";
}

bool ExactKnnClassifier::BetterNeighbour::operator()(
    const Neighbour& left, const Neighbour& right) const noexcept {
    if (left.squared_distance != right.squared_distance) {
        return left.squared_distance < right.squared_distance;
    }
    return left.index < right.index;
}

ExactKnnClassifier::ExactKnnClassifier(Matrix training,
                                       std::vector<int> labels,
                                       std::size_t k, WeightMode weights,
                                       std::optional<double> threshold)
    : training_(std::move(training)), labels_(std::move(labels)), k_(k),
      weights_(weights), threshold_(threshold) {
    validate_matrix_storage(training_, "training matrix", false);
    if (labels_.size() != training_.rows) {
        throw std::invalid_argument("training labels must match training rows");
    }
    if (k_ == 0 || k_ > training_.rows) {
        throw std::invalid_argument("k must be in [1, training rows]");
    }
    if (threshold_ && (!std::isfinite(*threshold_) || *threshold_ < 0.0 ||
                       *threshold_ > 1.0)) {
        throw std::invalid_argument("threshold must be finite and in [0, 1]");
    }
    for (int label : labels_) {
        if (label != 0 && label != 1) {
            throw std::invalid_argument("training labels must be binary 0 or 1");
        }
    }

}

void ExactKnnClassifier::validate_queries(const Matrix& queries) const {
    validate_matrix_storage(queries, "query matrix", true);
    if (queries.columns != training_.columns) {
        throw std::invalid_argument(
            "query feature count must match the training feature count"
        );
    }
}

std::vector<ExactKnnClassifier::Neighbour>
ExactKnnClassifier::reference_neighbours(std::span<const double> query) const {
    std::vector<Neighbour> candidates;
    candidates.reserve(training_.rows);
    for (std::size_t row = 0; row < training_.rows; ++row) {
        double squared_distance = 0.0;
        const auto training_row = training_.row(row);
        for (std::size_t feature = 0; feature < training_.columns; ++feature) {
            const double difference = query[feature] - training_row[feature];
            squared_distance += difference * difference;
        }
        if (!std::isfinite(squared_distance)) {
            throw std::overflow_error("non-finite squared Euclidean distance");
        }
        candidates.push_back({squared_distance, row});
    }
    std::sort(candidates.begin(), candidates.end(), BetterNeighbour{});
    candidates.resize(k_);
    return candidates;
}

std::vector<ExactKnnClassifier::Neighbour>
ExactKnnClassifier::nth_neighbours(const std::vector<double>& distances,
                                   std::size_t query_index,
                                   std::size_t query_count,
                                   std::vector<Neighbour>& candidates) const {
    candidates.resize(training_.rows);
    for (std::size_t row = 0; row < training_.rows; ++row) {
        candidates[row] = {
            distances[row * query_count + query_index], row
        };
    }
    if (k_ < candidates.size()) {
        std::nth_element(
            candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(k_),
            candidates.end(), BetterNeighbour{}
        );
        candidates.resize(k_);
    }
    std::sort(candidates.begin(), candidates.end(), BetterNeighbour{});
    return candidates;
}

PredictionResult ExactKnnClassifier::predict_reference(const Matrix& queries) const {
    validate_queries(queries);
    PredictionResult result;
    result.labels.resize(queries.rows);
    result.positive_vote_counts.resize(queries.rows);
    result.scores.resize(queries.rows);
    const auto prediction_start = Clock::now();
    for (std::size_t row = 0; row < queries.rows; ++row) {
        const std::vector<Neighbour> neighbours =
            reference_neighbours(queries.row(row));
        vote(neighbours, result, row);
    }
    result.distance_seconds = elapsed_seconds(prediction_start);
    return result;
}

void ExactKnnClassifier::vote(const std::vector<Neighbour>& neighbours,
                             PredictionResult& output, std::size_t row) const {
    std::size_t positive = 0, zeros = 0, zero_positive = 0;
    for (const auto& neighbour : neighbours) {
        const auto label = static_cast<std::size_t>(labels_[neighbour.index]);
        positive += label;
        if (neighbour.squared_distance == 0.0) {
            ++zeros;
            zero_positive += label;
        }
    }
    double score = static_cast<double>(positive) / static_cast<double>(k_);
    if (weights_ == WeightMode::distance) {
        if (zeros) {
            score = static_cast<double>(zero_positive) / static_cast<double>(zeros);
        } else {
            double sum = 0.0, positive_sum = 0.0;
            for (const auto& neighbour : neighbours) {
                const double weight = 1.0 / std::sqrt(neighbour.squared_distance);
                if (!std::isfinite(weight)) {
                    throw std::overflow_error("non-finite inverse-distance weight");
                }
                sum += weight;
                if (labels_[neighbour.index] == 1) positive_sum += weight;
            }
            if (!std::isfinite(sum) || !std::isfinite(positive_sum) || sum <= 0.0) {
                throw std::overflow_error("non-finite inverse-distance vote sum");
            }
            score = positive_sum / sum;
        }
    }
    if (!std::isfinite(score) || score < 0.0 || score > 1.0) {
        throw std::overflow_error("invalid class-1 probability");
    }
    output.positive_vote_counts[row] = positive;
    output.scores[row] = score;
    output.labels[row] = threshold_ ? static_cast<int>(score >= *threshold_)
                                    : static_cast<int>(score > 0.5);
}

PredictionResult ExactKnnClassifier::predict_optimized(
    const Matrix& queries, std::size_t batch_size,
    SelectionMethod selection_method) const {
    validate_queries(queries);
    if (batch_size == 0) {
        throw std::invalid_argument("batch size must be at least 1");
    }
    PredictionResult output;
    output.labels.resize(queries.rows);
    output.positive_vote_counts.resize(queries.rows);
    output.scores.resize(queries.rows);
    if (queries.rows == 0) {
        return output;
    }

    const std::size_t reserved_batch = std::min(batch_size, queries.rows);
    std::vector<double> distances;
    std::vector<Neighbour> nth_candidates;
    if (selection_method == SelectionMethod::nth_element) {
        if (training_.rows >
            std::numeric_limits<std::size_t>::max() / reserved_batch) {
            throw std::length_error("distance workspace dimensions overflow");
        }
        distances.resize(training_.rows * reserved_batch);
        nth_candidates.resize(training_.rows);
    }
    std::vector<double> query_feature_major(training_.columns * reserved_batch);
    std::vector<double> row_distances(reserved_batch);
    std::vector<std::vector<Neighbour>> selected(reserved_batch);
    for (auto& neighbours : selected) neighbours.reserve(k_);

    for (std::size_t start = 0; start < queries.rows; start += batch_size) {
        const std::size_t count = std::min(batch_size, queries.rows - start);
        const auto distance_start = Clock::now();
        for (std::size_t feature = 0; feature < training_.columns; ++feature) {
            for (std::size_t query = 0; query < count; ++query) {
                query_feature_major[feature * count + query] =
                    queries.values[(start + query) * queries.columns + feature];
            }
        }
        for (std::size_t query = 0; query < count; ++query) {
            selected[query].clear();
        }

        // A training row is consumed once per query batch. The heap path
        // immediately folds each exact distance into its bounded top-k heap,
        // avoiding a full distance matrix and a later strided scan.
        for (std::size_t row = 0; row < training_.rows; ++row) {
            std::fill(row_distances.begin(), row_distances.begin() + count, 0.0);
            const double* training_row =
                training_.values.data() + row * training_.columns;
            for (std::size_t feature = 0; feature < training_.columns; ++feature) {
                const double training_value = training_row[feature];
                const double* query_feature =
                    query_feature_major.data() + feature * count;
                for (std::size_t query = 0; query < count; ++query) {
                    const double difference = query_feature[query] - training_value;
                    row_distances[query] += difference * difference;
                }
            }
            for (std::size_t query = 0; query < count; ++query) {
                if (!std::isfinite(row_distances[query])) {
                    throw std::overflow_error("non-finite squared Euclidean distance");
                }
            }
            if (selection_method == SelectionMethod::heap) {
                for (std::size_t query = 0; query < count; ++query) {
                    auto& best = selected[query];
                    const Neighbour candidate{row_distances[query], row};
                    if (best.size() < k_) {
                        best.push_back(candidate);
                        std::push_heap(best.begin(), best.end(), BetterNeighbour{});
                    } else if (BetterNeighbour{}(candidate, best.front())) {
                        std::pop_heap(best.begin(), best.end(), BetterNeighbour{});
                        best.back() = candidate;
                        std::push_heap(best.begin(), best.end(), BetterNeighbour{});
                    }
                }
            } else {
                std::copy(row_distances.begin(), row_distances.begin() + count,
                          distances.begin() + static_cast<std::ptrdiff_t>(row * count));
            }
        }
        output.distance_seconds += elapsed_seconds(distance_start);

        const auto selection_start = Clock::now();
        if (selection_method == SelectionMethod::heap) {
            for (std::size_t query = 0; query < count; ++query) {
                std::sort(selected[query].begin(), selected[query].end(),
                          BetterNeighbour{});
            }
        } else {
            for (std::size_t query = 0; query < count; ++query) {
                selected[query] =
                    nth_neighbours(distances, query, count, nth_candidates);
            }
        }
        for (std::size_t query = 0; query < count; ++query) {
            vote(selected[query], output, start + query);
        }
        output.selection_vote_seconds += elapsed_seconds(selection_start);
    }
    return output;
}

std::size_t ExactKnnClassifier::training_rows() const noexcept {
    return training_.rows;
}

std::size_t ExactKnnClassifier::feature_count() const noexcept {
    return training_.columns;
}

std::size_t ExactKnnClassifier::k() const noexcept {
    return k_;
}

std::size_t ExactKnnClassifier::auxiliary_bytes() const noexcept {
    return 0;
}

}  // namespace cpp_knn
