#include "cpp_knn/data_io.hpp"
#include "cpp_knn/knn.hpp"

#include <filesystem>
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
        std::cout << "cpp_knn_tests: all tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cpp_knn_tests failed: " << error.what() << '\n';
        return 1;
    }
}
