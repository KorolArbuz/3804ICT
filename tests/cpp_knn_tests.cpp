#include "cpp_knn/data_io.hpp"
#include "cpp_knn/knn.hpp"

#include <filesystem>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cpp_knn::ExactKnnClassifier;
using cpp_knn::Matrix;
using cpp_knn::PredictionResult;
using cpp_knn::SelectionMethod;
using cpp_knn::WeightMode;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void require_throws(Function function, const std::string& message) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

void require_equal(const PredictionResult& left, const PredictionResult& right,
                   const std::string& name) {
    require(left.labels == right.labels, name + ": labels differ");
    require(left.positive_vote_counts == right.positive_vote_counts,
            name + ": vote counts differ");
    require(left.scores == right.scores, name + ": scores differ");
}

Matrix matrix(std::size_t rows, std::size_t columns,
              std::initializer_list<double> values) {
    return Matrix{rows, columns, std::vector<double>(values)};
}

void test_tiny_and_k_one() {
    ExactKnnClassifier classifier(
        matrix(4, 2, {0, 0, 1, 0, 0, 2, 3, 3}), {0, 1, 1, 0}, 1
    );
    const Matrix queries = matrix(3, 2, {0.1, 0.0, 0.9, 0.0, 0.0, 1.8});
    const auto expected = classifier.predict_reference(queries);
    require(expected.labels == std::vector<int>({0, 1, 1}),
            "hand-calculated k=1 predictions differ");
    require_equal(expected,
                  classifier.predict_optimized(queries, 2, SelectionMethod::heap),
                  "tiny heap");
    require_equal(expected,
                  classifier.predict_optimized(queries, 2,
                                               SelectionMethod::nth_element),
                  "tiny nth_element");
}

void test_duplicates_ties_and_repeated_queries() {
    const Matrix training = matrix(
        8, 2,
        {-1, 0, 1, 0, 0, -1, 0, 1, -1, 0, 1, 0, 2, 2, -2, -2}
    );
    ExactKnnClassifier classifier(training, {1, 0, 1, 0, 0, 1, 1, 0}, 4);
    const Matrix queries = matrix(3, 2, {0, 0, 0, 0, -1, 0});
    const auto reference = classifier.predict_reference(queries);
    require(reference.positive_vote_counts[0] == 2,
            "equal-distance index tie chose wrong labels");
    require(reference.labels[0] == 0,
            "even-k class tie must prefer class zero");
    require(reference.positive_vote_counts[0] ==
                reference.positive_vote_counts[1] &&
            reference.labels[0] == reference.labels[1],
            "identical queries changed results");
    require_equal(reference,
                  classifier.predict_optimized(queries, 3, SelectionMethod::heap),
                  "duplicate heap");
    require_equal(reference,
                  classifier.predict_optimized(queries, 3,
                                               SelectionMethod::nth_element),
                  "duplicate nth_element");
}

void test_project_k_and_single_class() {
    std::vector<double> values;
    values.reserve(40 * 3);
    for (std::size_t row = 0; row < 40; ++row) {
        values.push_back(static_cast<double>(row));
        values.push_back(static_cast<double>(row % 7));
        values.push_back(-static_cast<double>(row % 5));
    }
    ExactKnnClassifier classifier(Matrix{40, 3, values},
                                  std::vector<int>(40, 1), 19);
    const Matrix queries = matrix(2, 3, {8.5, 1, -2, 31.5, 4, -1});
    const auto reference = classifier.predict_reference(queries);
    require(reference.labels == std::vector<int>({1, 1}),
            "single-class labels changed");
    require(reference.positive_vote_counts == std::vector<std::size_t>({19, 19}),
            "single-class vote counts changed");
    for (std::size_t batch : {1U, 2U, 32U}) {
        require_equal(reference,
                      classifier.predict_optimized(queries, batch,
                                                   SelectionMethod::heap),
                      "project k heap");
    }
}

void test_scalar_optimized_random_equivalence() {
    std::mt19937_64 random(3804);
    std::uniform_real_distribution<double> distribution(-100.0, 100.0);
    std::vector<double> training_values(73 * 11);
    std::vector<double> query_values(17 * 11);
    for (double& value : training_values) value = distribution(random);
    for (double& value : query_values) value = distribution(random);
    std::vector<int> labels(73);
    for (std::size_t index = 0; index < labels.size(); ++index) {
        labels[index] = static_cast<int>((index * 7) % 2);
    }
    ExactKnnClassifier classifier(Matrix{73, 11, training_values}, labels, 19);
    const Matrix queries{17, 11, query_values};
    const auto reference = classifier.predict_reference(queries);
    for (std::size_t batch : {1U, 4U, 8U, 64U}) {
        require_equal(reference,
                      classifier.predict_optimized(queries, batch,
                                                   SelectionMethod::heap),
                      "random heap");
        require_equal(reference,
                      classifier.predict_optimized(
                          queries, batch, SelectionMethod::nth_element),
                      "random nth_element");
    }
    const auto repeated = classifier.predict_optimized(
        queries, 8, SelectionMethod::heap
    );
    require_equal(repeated,
                  classifier.predict_optimized(queries, 8, SelectionMethod::heap),
                  "repeated execution");
}

void test_validation_and_empty_query_policy() {
    const Matrix valid = matrix(3, 2, {0, 0, 1, 1, 2, 2});
    require_throws(
        [&] { (void)ExactKnnClassifier(valid, {0, 1, 0}, 0); },
        "k=0 was accepted"
    );
    require_throws(
        [&] { (void)ExactKnnClassifier(valid, {0, 1, 0}, 4); },
        "k greater than training rows was accepted"
    );
    require_throws(
        [&] { (void)ExactKnnClassifier(valid, {0, 1}, 1); },
        "mismatched labels were accepted"
    );
    require_throws(
        [&] { (void)ExactKnnClassifier(Matrix{0, 2, {}}, {}, 1); },
        "empty training data was accepted"
    );
    require_throws(
        [&] {
            (void)ExactKnnClassifier(
                Matrix{1, 2, {0, std::numeric_limits<double>::quiet_NaN()}},
                {0}, 1
            );
        },
        "NaN training value was accepted"
    );
    ExactKnnClassifier classifier(valid, {0, 1, 0}, 1);
    require_throws(
        [&] { (void)classifier.predict_optimized(Matrix{1, 3, {0, 0, 0}}, 1,
                                                 SelectionMethod::heap); },
        "mismatched query dimensions were accepted"
    );
    require_throws(
        [&] {
            (void)classifier.predict_optimized(
                Matrix{1, 2, {0, std::numeric_limits<double>::infinity()}}, 1,
                SelectionMethod::heap
            );
        },
        "infinite query value was accepted"
    );
    const auto empty = classifier.predict_optimized(
        Matrix{0, 2, {}}, 1, SelectionMethod::heap
    );
    require(empty.labels.empty() && empty.positive_vote_counts.empty(),
            "empty query matrix should return empty output like V5.1");
}

void test_standard_library_csv_loader_and_hash() {
    const auto directory = std::filesystem::temp_directory_path();
    const auto valid_path = directory / "cpp_knn_test_features.csv";
    const auto labels_path = directory / "cpp_knn_test_labels.csv";
    const auto invalid_path = directory / "cpp_knn_test_invalid.csv";
    {
        std::ofstream output(valid_path);
        output << "1.25,-2\n3,4.5\n";
    }
    {
        std::ofstream output(labels_path);
        output << "0\n1\n";
    }
    {
        std::ofstream output(invalid_path);
        output << "1,nan\n";
    }
    const Matrix loaded = cpp_knn::load_feature_csv(valid_path);
    require(loaded.rows == 2 && loaded.columns == 2 && loaded.values[1] == -2.0,
            "CSV matrix loader changed values");
    require(cpp_knn::load_label_csv(labels_path) == std::vector<int>({0, 1}),
            "CSV label loader changed values");
    require_throws([&] { (void)cpp_knn::load_feature_csv(invalid_path); },
                   "CSV loader accepted NaN");
    const std::string hash = cpp_knn::prediction_hash({0, 1}, {3, 7});
    require(hash == cpp_knn::prediction_hash({0, 1}, {3, 7}),
            "prediction hash is not deterministic");
    std::filesystem::remove(valid_path);
    std::filesystem::remove(labels_path);
    std::filesystem::remove(invalid_path);
}

void test_inverse_distance_and_zero_semantics() {
    const Matrix queries = matrix(1, 1, {0});
    ExactKnnClassifier weighted(matrix(2, 1, {1, 2}), {1, 0}, 2,
                                WeightMode::distance, 0.6);
    const auto score = weighted.predict_reference(queries);
    require(std::abs(score.scores[0] - 2.0 / 3.0) < 1e-15,
            "inverse distance was replaced by inverse squared distance");
    require(score.labels[0] == 1, "distance threshold label differs");
    require_equal(score, weighted.predict_optimized(queries, 32, SelectionMethod::heap),
                  "weighted hand calculation");
    ExactKnnClassifier zero(matrix(4, 1, {0, 0, 0, 1}), {1, 0, 1, 1}, 4,
                            WeightMode::distance, 0.7);
    require(zero.predict_reference(queries).scores[0] == 2.0 / 3.0,
            "nonzero neighbour participated with exact zero neighbours");
    ExactKnnClassifier duplicates(matrix(5, 1, {0, 0, 0, 0, 0}), {1, 0, 0, 1, 1}, 2,
                                  WeightMode::distance, 0.5);
    const auto duplicate_result = duplicates.predict_optimized(queries, 32, SelectionMethod::heap);
    require(duplicate_result.scores[0] == 0.5 && duplicate_result.labels[0] == 1,
            "more-than-k duplicates did not select first original indices");
    require_equal(duplicate_result, duplicates.predict_reference(queries), "zero duplicates");
}

void test_threshold_boundary_and_fit_copy() {
    Matrix training = matrix(2, 1, {-1, 1});
    std::vector<int> labels{0, 1};
    const Matrix queries = matrix(1, 1, {0});
    ExactKnnClassifier original(training, labels, 2);
    require(original.predict_reference(queries).labels[0] == 0,
            "uniform default argmax tie did not choose 0");
    ExactKnnClassifier equal(training, labels, 2, WeightMode::distance, 0.5);
    ExactKnnClassifier below(training, labels, 2, WeightMode::distance,
                            std::nextafter(0.5, 0.0));
    ExactKnnClassifier above(training, labels, 2, WeightMode::distance,
                            std::nextafter(0.5, 1.0));
    training.values[0] = 100;
    labels[0] = 1;
    require(equal.predict_reference(queries).scores[0] == 0.5,
            "constructor did not own a copy of fit data");
    require(equal.predict_reference(queries).labels[0] == 1 &&
            below.predict_reference(queries).labels[0] == 1 &&
            above.predict_reference(queries).labels[0] == 0,
            "exact/nextafter threshold decision differs");
}

void test_weighted_oracle_multiple_k_dimensions() {
    std::mt19937_64 generator(101);
    std::uniform_real_distribution<double> random(-3.0, 3.0);
    for (const std::size_t columns : {1U, 2U, 17U, 65U}) {
        Matrix training{109, columns, std::vector<double>(109 * columns)};
        Matrix queries{35, columns, std::vector<double>(35 * columns)};
        std::vector<int> labels(109);
        for (double& value : training.values) value = random(generator);
        for (double& value : queries.values) value = random(generator);
        for (std::size_t row = 0; row < labels.size(); ++row) labels[row] = static_cast<int>(row % 2);
        for (const std::size_t k : {1U, 19U, 101U, 109U}) {
            ExactKnnClassifier model(training, labels, k, WeightMode::distance, 0.3315411365543412);
            const auto expected = model.predict_reference(queries);
            for (const std::size_t batch : {1U, 8U, 32U}) {
                require_equal(expected, model.predict_optimized(queries, batch, SelectionMethod::heap),
                              "weighted heap reference");
                require_equal(expected, model.predict_optimized(queries, batch, SelectionMethod::nth_element),
                              "weighted nth reference");
            }
        }
    }
}

void test_weighted_boundary_and_numerical_errors() {
    ExactKnnClassifier boundary(matrix(4, 1, {-1, 1, -1, 1}), {1, 0, 0, 1}, 3,
                                WeightMode::distance, 0.5);
    const Matrix query = matrix(1, 1, {0});
    const auto result = boundary.predict_reference(query);
    require(result.scores[0] == 1.0 / 3.0, "boundary tie ignored original training indices");
    require_equal(result, boundary.predict_optimized(query, 32, SelectionMethod::heap),
                  "weighted boundary tie");
    for (double bad : {-0.1, 1.1, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
        require_throws([&] { (void)ExactKnnClassifier(matrix(1, 1, {0}), {0}, 1,
                                                    WeightMode::distance, bad); },
                       "invalid threshold was accepted");
    }
    ExactKnnClassifier huge(matrix(1, 1, {1e308}), {0}, 1, WeightMode::distance, 0.5);
    require_throws([&] { (void)huge.predict_reference(query); }, "reference silently accepted distance overflow");
    require_throws([&] { (void)huge.predict_optimized(query, 32, SelectionMethod::heap); },
                   "heap silently accepted distance overflow");
    require_throws([&] { (void)huge.predict_optimized(query, 32, SelectionMethod::nth_element); },
                   "nth silently accepted distance overflow");
    require_throws([&] { (void)boundary.predict_optimized(query, 0, SelectionMethod::heap); },
                   "zero batch size accepted");
    require_throws([&] { (void)cpp_knn::parse_weight_mode("squared"); }, "invalid weighting accepted");
    require_throws([&] { (void)ExactKnnClassifier(matrix(1, 1, {0}), {2}, 1); }, "nonbinary label accepted");
    const auto empty = boundary.predict_optimized(Matrix{0, 1, {}}, 32, SelectionMethod::heap);
    require(empty.scores.empty() && empty.labels.empty(), "empty final query policy differs");
}

}  // namespace

int main() {
    try {
        std::cerr << "running tiny/k=1\n";
        test_tiny_and_k_one();
        std::cerr << "running duplicates/ties\n";
        test_duplicates_ties_and_repeated_queries();
        std::cerr << "running project k/single class\n";
        test_project_k_and_single_class();
        std::cerr << "running randomized equivalence\n";
        test_scalar_optimized_random_equivalence();
        std::cerr << "running validation/empty query\n";
        test_validation_and_empty_query_policy();
        std::cerr << "running CSV/hash\n";
        test_standard_library_csv_loader_and_hash();
        std::cerr << "running inverse distance / zero semantics\n";
        test_inverse_distance_and_zero_semantics();
        std::cerr << "running threshold boundary / immutable fit\n";
        test_threshold_boundary_and_fit_copy();
        std::cerr << "running weighted k=1/19/101/n independent sort oracle\n";
        test_weighted_oracle_multiple_k_dimensions();
        std::cerr << "running weighted boundary / numerical errors\n";
        test_weighted_boundary_and_numerical_errors();
        std::cout << "cpp_knn_tests: all tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cpp_knn_tests failed: " << error.what() << '\n';
        return 1;
    }
}
