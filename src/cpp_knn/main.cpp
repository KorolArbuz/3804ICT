#include "cpp_knn/data_io.hpp"
#include "cpp_knn/knn.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Arguments {
    std::filesystem::path train_features;
    std::filesystem::path train_labels;
    std::filesystem::path test_features;
    std::filesystem::path test_labels;
    std::filesystem::path output;
    std::size_t k{};
    std::size_t warmups{1};
    std::size_t runs{15};
    std::size_t batch_size{32};
    cpp_knn::SelectionMethod selection{cpp_knn::SelectionMethod::heap};
    bool reference{};
};

[[nodiscard]] std::string compiler_name() {
#if defined(_MSC_VER)
    return "MSVC _MSC_VER=" + std::to_string(_MSC_VER) +
           " _MSC_FULL_VER=" + std::to_string(_MSC_FULL_VER);
#elif defined(__clang__)
    return "Clang " __clang_version__;
#elif defined(__GNUC__)
    return "GCC " __VERSION__;
#else
    return "unknown";
#endif
}

[[nodiscard]] std::string build_mode() {
#ifdef NDEBUG
    constexpr std::string_view mode = "Release";
#else
    constexpr std::string_view mode = "Debug";
#endif
#ifdef CPP_KNN_NATIVE_BUILD
    return std::string(mode) + " native";
#else
    return std::string(mode) + " portable";
#endif
}

[[nodiscard]] std::size_t parse_size(const std::string& text,
                                     std::string_view option,
                                     bool allow_zero = false) {
    std::size_t consumed = 0;
    unsigned long long value = 0;
    try {
        value = std::stoull(text, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(option) + " must be an integer");
    }
    if (consumed != text.size() || (!allow_zero && value == 0)) {
        throw std::invalid_argument(std::string(option) + " has an invalid value");
    }
    return static_cast<std::size_t>(value);
}

void print_help() {
    std::cout
        << "Pure C++20 exact exhaustive KNN\n\n"
        << "cpp_knn --train-features <csv> --train-labels <csv> "
           "--test-features <csv> --test-labels <csv> --k <integer> "
           "--output <json> [--warmups N] [--runs N] [--batch-size N] "
           "[--selection heap|nth] [--algorithm optimized|reference]\n";
}

[[nodiscard]] Arguments parse_arguments(int argc, char** argv) {
    if (argc == 1) {
        print_help();
        throw std::invalid_argument("arguments are required");
    }
    std::map<std::string, std::string> values;
    for (int index = 1; index < argc; index += 2) {
        const std::string key(argv[index]);
        if (key == "--help") {
            print_help();
            std::exit(0);
        }
        if (index + 1 >= argc || !key.starts_with("--") ||
            !values.emplace(key, argv[index + 1]).second) {
            throw std::invalid_argument("invalid, duplicate, or incomplete option: " + key);
        }
    }
    const auto required = [&values](std::string_view name) -> const std::string& {
        const auto found = values.find(std::string(name));
        if (found == values.end()) {
            throw std::invalid_argument("required option: " + std::string(name));
        }
        return found->second;
    };
    const std::vector<std::string> allowed{
        "--train-features", "--train-labels", "--test-features",
        "--test-labels", "--k", "--output", "--warmups", "--runs",
        "--batch-size", "--selection", "--algorithm"
    };
    for (const auto& [key, unused] : values) {
        (void)unused;
        if (std::find(allowed.begin(), allowed.end(), key) == allowed.end()) {
            throw std::invalid_argument("unknown option: " + key);
        }
    }

    Arguments arguments;
    arguments.train_features = required("--train-features");
    arguments.train_labels = required("--train-labels");
    arguments.test_features = required("--test-features");
    arguments.test_labels = required("--test-labels");
    arguments.output = required("--output");
    arguments.k = parse_size(required("--k"), "--k");
    if (values.contains("--warmups")) {
        arguments.warmups = parse_size(values.at("--warmups"), "--warmups", true);
    }
    if (values.contains("--runs")) {
        arguments.runs = parse_size(values.at("--runs"), "--runs");
    }
    if (values.contains("--batch-size")) {
        arguments.batch_size = parse_size(values.at("--batch-size"), "--batch-size");
    }
    if (values.contains("--selection")) {
        arguments.selection =
            cpp_knn::parse_selection_method(values.at("--selection"));
    }
    if (values.contains("--algorithm")) {
        const std::string& algorithm = values.at("--algorithm");
        if (algorithm != "optimized" && algorithm != "reference") {
            throw std::invalid_argument("--algorithm must be optimized or reference");
        }
        arguments.reference = algorithm == "reference";
    }
    return arguments;
}

[[nodiscard]] double percentile(std::vector<double> values, double probability) {
    if (values.empty()) {
        throw std::invalid_argument("cannot summarize empty timings");
    }
    std::sort(values.begin(), values.end());
    const double position = probability * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return values[lower] + fraction * (values[upper] - values[lower]);
}

void write_number_array(std::ostream& output, const std::vector<double>& values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) output << ',';
        output << values[index];
    }
    output << ']';
}

template <typename T>
void write_integer_array(std::ostream& output, const std::vector<T>& values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) output << ',';
        output << values[index];
    }
    output << ']';
}

void write_result(const Arguments& arguments,
                  const cpp_knn::ExactKnnClassifier& classifier,
                  const std::vector<int>& test_labels,
                  const cpp_knn::PredictionResult& predictions,
                  const std::vector<double>& raw_seconds,
                  const std::vector<double>& raw_distance_seconds,
                  const std::vector<double>& raw_selection_seconds,
                  double fit_seconds) {
    std::size_t tn = 0, fp = 0, fn = 0, tp = 0;
    for (std::size_t index = 0; index < test_labels.size(); ++index) {
        if (test_labels[index] == 0 && predictions.labels[index] == 0) ++tn;
        if (test_labels[index] == 0 && predictions.labels[index] == 1) ++fp;
        if (test_labels[index] == 1 && predictions.labels[index] == 0) ++fn;
        if (test_labels[index] == 1 && predictions.labels[index] == 1) ++tp;
    }
    const std::size_t bounded_batch = std::min(arguments.batch_size, test_labels.size());
    const std::size_t neighbour_bytes = sizeof(double) + sizeof(std::size_t);
    const std::size_t selected_neighbour_bytes =
        bounded_batch * classifier.k() * neighbour_bytes;
    const std::size_t selector_bytes =
        selected_neighbour_bytes +
        bounded_batch * sizeof(std::vector<std::size_t>) +
        (arguments.selection == cpp_knn::SelectionMethod::nth_element
             ? classifier.training_rows() * neighbour_bytes
             : 0);
    const std::size_t distance_bytes =
        arguments.selection == cpp_knn::SelectionMethod::nth_element
        ? classifier.training_rows() * bounded_batch * sizeof(double)
        : bounded_batch * sizeof(double);
    const std::size_t peak_auxiliary_bytes =
        classifier.auxiliary_bytes() +
        distance_bytes +
        classifier.feature_count() * bounded_batch * sizeof(double) +
        selector_bytes +
        test_labels.size() * (sizeof(int) + sizeof(std::size_t));

    if (!arguments.output.parent_path().empty()) {
        std::filesystem::create_directories(arguments.output.parent_path());
    }
    std::ofstream output(arguments.output);
    if (!output) {
        throw std::runtime_error("cannot open output JSON: " + arguments.output.string());
    }
    output << std::setprecision(17);
    output << "{\n"
           << "  \"implementation\": \"custom_cpp_exact_knn\",\n"
           << "  \"algorithm\": \"" << (arguments.reference ? "scalar_reference" : "optimized") << "\",\n"
           << "  \"selection\": \"" << cpp_knn::selection_method_name(arguments.selection) << "\",\n"
           << "  \"training_rows\": " << classifier.training_rows() << ",\n"
           << "  \"test_rows\": " << test_labels.size() << ",\n"
           << "  \"feature_count\": " << classifier.feature_count() << ",\n"
           << "  \"k\": " << classifier.k() << ",\n"
           << "  \"batch_size\": " << arguments.batch_size << ",\n"
           << "  \"warmup_count\": " << arguments.warmups << ",\n"
           << "  \"measured_run_count\": " << arguments.runs << ",\n"
           << "  \"fit_seconds\": " << fit_seconds << ",\n"
           << "  \"prediction_seconds\": ";
    write_number_array(output, raw_seconds);
    output << ",\n  \"distance_or_fused_heap_phase_seconds\": ";
    write_number_array(output, raw_distance_seconds);
    output << ",\n  \"selection_vote_phase_seconds\": ";
    write_number_array(output, raw_selection_seconds);
    output << ",\n"
           << "  \"median_prediction_seconds\": " << percentile(raw_seconds, 0.5) << ",\n"
           << "  \"min_prediction_seconds\": " << *std::min_element(raw_seconds.begin(), raw_seconds.end()) << ",\n"
           << "  \"max_prediction_seconds\": " << *std::max_element(raw_seconds.begin(), raw_seconds.end()) << ",\n"
           << "  \"iqr_prediction_seconds\": " << percentile(raw_seconds, 0.75) - percentile(raw_seconds, 0.25) << ",\n"
           << "  \"prediction_count\": " << predictions.labels.size() << ",\n"
           << "  \"predicted_labels\": ";
    write_integer_array(output, predictions.labels);
    output << ",\n  \"positive_vote_counts\": ";
    write_integer_array(output, predictions.positive_vote_counts);
    output << ",\n  \"positive_vote_fractions\": [";
    for (std::size_t index = 0; index < predictions.positive_vote_counts.size(); ++index) {
        if (index) output << ',';
        output << static_cast<double>(predictions.positive_vote_counts[index]) /
                      static_cast<double>(classifier.k());
    }
    output << "],\n"
           << "  \"prediction_hash\": \""
           << cpp_knn::prediction_hash(predictions.labels,
                                       predictions.positive_vote_counts) << "\",\n"
           << "  \"confusion_matrix\": {\"TN\":" << tn << ",\"FP\":" << fp
           << ",\"FN\":" << fn << ",\"TP\":" << tp << "},\n"
           << "  \"compiler\": \"" << compiler_name() << "\",\n"
           << "  \"cpp_standard\": 20,\n"
           << "  \"build_mode\": \"" << build_mode() << "\",\n"
           << "  \"thread_count\": 1,\n"
           << "  \"peak_auxiliary_bytes_estimate\": " << peak_auxiliary_bytes << ",\n"
           << "  \"timing_scope\": \"exact distances, selection, uniform voting, and output-vector construction; input parsing, model fit, warm-up, process startup, and JSON serialization excluded\"\n"
           << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        cpp_knn::Matrix training = cpp_knn::load_feature_csv(arguments.train_features);
        std::vector<int> training_labels = cpp_knn::load_label_csv(arguments.train_labels);
        cpp_knn::Matrix testing = cpp_knn::load_feature_csv(arguments.test_features);
        std::vector<int> test_labels = cpp_knn::load_label_csv(arguments.test_labels);
        if (testing.rows != test_labels.size()) {
            throw std::invalid_argument("test labels must match test feature rows");
        }

        const auto fit_start = Clock::now();
        cpp_knn::ExactKnnClassifier classifier(
            std::move(training), std::move(training_labels), arguments.k
        );
        const double fit_seconds =
            std::chrono::duration<double>(Clock::now() - fit_start).count();

        const auto predict = [&]() {
            if (arguments.reference) {
                return classifier.predict_reference(testing);
            }
            return classifier.predict_optimized(
                testing, arguments.batch_size, arguments.selection
            );
        };
        for (std::size_t warmup = 0; warmup < arguments.warmups; ++warmup) {
            (void)predict();
        }

        std::vector<double> raw_seconds;
        std::vector<double> raw_distance_seconds;
        std::vector<double> raw_selection_seconds;
        raw_seconds.reserve(arguments.runs);
        cpp_knn::PredictionResult final_predictions;
        for (std::size_t run = 0; run < arguments.runs; ++run) {
            const auto start = Clock::now();
            cpp_knn::PredictionResult predictions = predict();
            raw_seconds.push_back(
                std::chrono::duration<double>(Clock::now() - start).count()
            );
            raw_distance_seconds.push_back(predictions.distance_seconds);
            raw_selection_seconds.push_back(predictions.selection_vote_seconds);
            if (run && (predictions.labels != final_predictions.labels ||
                        predictions.positive_vote_counts !=
                            final_predictions.positive_vote_counts)) {
                throw std::runtime_error("predictions changed between measured runs");
            }
            final_predictions = std::move(predictions);
        }
        write_result(arguments, classifier, test_labels, final_predictions,
                     raw_seconds, raw_distance_seconds, raw_selection_seconds,
                     fit_seconds);
        std::cout << "custom_cpp_exact_knn: k=" << classifier.k() << ", "
                  << final_predictions.labels.size() << " predictions, median "
                  << std::fixed << std::setprecision(6)
                  << percentile(raw_seconds, 0.5) << "s -> "
                  << arguments.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cpp_knn error: " << error.what() << '\n';
        return 1;
    }
}
