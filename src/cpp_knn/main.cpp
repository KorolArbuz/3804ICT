#include "cpp_knn/data_io.hpp"
#include "cpp_knn/knn.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
struct Arguments {
    std::filesystem::path train_features, train_labels, test_features, test_labels;
    std::filesystem::path output, predictions;
    std::size_t k{}, warmups{1}, runs{1}, batch_size{32};
    cpp_knn::SelectionMethod selection{cpp_knn::SelectionMethod::heap};
    cpp_knn::WeightMode weights{cpp_knn::WeightMode::uniform};
    std::optional<double> threshold;
    std::string model_id{"custom_cpp"}, config_id{"unspecified"};
    bool reference{}, persistent{};
};

std::string quoted(const std::string& text) {
    std::ostringstream output;
    output << '"';
    for (const unsigned char character : text) {
        if (character == '"' || character == '\\') output << '\\' << character;
        else if (character < 32) output << "\\u" << std::hex << std::setw(4)
                                       << std::setfill('0') << static_cast<int>(character);
        else output << character;
    }
    output << '"';
    return output.str();
}

std::string compiler_name() {
#if defined(_MSC_VER)
    return "MSVC " + std::to_string(_MSC_FULL_VER);
#elif defined(__clang__)
    return "Clang " __clang_version__;
#elif defined(__GNUC__)
    return "GCC " __VERSION__;
#else
    return "unknown";
#endif
}
std::string build_mode() {
#ifdef CPP_KNN_NATIVE_BUILD
    return "native";
#else
    return "portable";
#endif
}

void identity(std::ostream& output) {
    output << "\"source_id\":" << quoted(CPP_KNN_SOURCE_ID)
           << ",\"compiler\":" << quoted(compiler_name())
           << ",\"build_mode\":" << quoted(build_mode())
#ifdef NDEBUG
           << ",\"build_configuration\":\"Release\""
#else
           << ",\"build_configuration\":\"Debug\""
#endif
           << ",\"flags\":" << quoted(CPP_KNN_BUILD_FLAGS)
#ifdef CPP_KNN_LTO_ENABLED
           << ",\"lto_enabled\":true"
#else
           << ",\"lto_enabled\":false"
#endif
           << ",\"cpp_standard\":20,\"thread_count\":1";
}

std::size_t parse_size(const std::string& value, bool allow_zero = false) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("size must contain only decimal digits");
    std::size_t consumed = 0;
    const auto parsed = std::stoull(value, &consumed);
    if (consumed != value.size() || (!allow_zero && parsed == 0) ||
        parsed > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("invalid size");
    return static_cast<std::size_t>(parsed);
}

Arguments parse_arguments(int argc, char** argv) {
    std::map<std::string, std::string> values;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc || !values.emplace(argv[index], argv[index + 1]).second)
            throw std::invalid_argument("option requires a unique key and value");
    }
    const std::vector<std::string> allowed{
        "--train-features", "--train-labels", "--test-features", "--test-labels",
        "--k", "--output", "--predictions", "--warmups", "--runs", "--batch-size",
        "--selection", "--algorithm", "--weights", "--threshold", "--model-id",
        "--config-id", "--persistent"};
    for (const auto& [key, value] : values) {
        (void)value;
        if (std::find(allowed.begin(), allowed.end(), key) == allowed.end())
            throw std::invalid_argument("unknown option: " + key);
    }
    const auto required = [&](const std::string& key) {
        if (!values.contains(key)) throw std::invalid_argument("required option: " + key);
        return values.at(key);
    };
    Arguments args;
    args.train_features = required("--train-features");
    args.train_labels = required("--train-labels");
    args.test_features = required("--test-features");
    args.k = parse_size(required("--k"));
    if (values.contains("--test-labels")) args.test_labels = values.at("--test-labels");
    if (values.contains("--output")) args.output = values.at("--output");
    if (values.contains("--predictions")) args.predictions = values.at("--predictions");
    if (values.contains("--warmups")) args.warmups = parse_size(values.at("--warmups"), true);
    if (values.contains("--runs")) args.runs = parse_size(values.at("--runs"));
    if (values.contains("--batch-size")) args.batch_size = parse_size(values.at("--batch-size"));
    if (values.contains("--selection")) args.selection = cpp_knn::parse_selection_method(values.at("--selection"));
    if (values.contains("--weights")) args.weights = cpp_knn::parse_weight_mode(values.at("--weights"));
    if (values.contains("--threshold")) {
        std::size_t consumed = 0;
        args.threshold = std::stod(values.at("--threshold"), &consumed);
        if (consumed != values.at("--threshold").size())
            throw std::invalid_argument("invalid threshold");
    }
    if (values.contains("--model-id")) args.model_id = values.at("--model-id");
    if (values.contains("--config-id")) args.config_id = values.at("--config-id");
    if (values.contains("--algorithm")) {
        const auto& algorithm = values.at("--algorithm");
        if (algorithm != "optimized" && algorithm != "reference")
            throw std::invalid_argument("algorithm must be optimized or reference");
        args.reference = algorithm == "reference";
    }
    if (values.contains("--persistent")) {
        const auto& value = values.at("--persistent");
        if (value != "true" && value != "false")
            throw std::invalid_argument("persistent must be true or false");
        args.persistent = value == "true";
    }
    if (!args.persistent && args.output.empty())
        throw std::invalid_argument("--output required outside persistent mode");
    return args;
}

template<class T> void array(std::ostream& output, const std::vector<T>& values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) output << ',';
        output << values[index];
    }
    output << ']';
}

double percentile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    const double position = q * static_cast<double>(values.size() - 1);
    const auto low = static_cast<std::size_t>(std::floor(position));
    const auto high = static_cast<std::size_t>(std::ceil(position));
    return values[low] + (position - static_cast<double>(low)) * (values[high] - values[low]);
}

// Binary64 scores and labels are hashed after stopping the prediction timer.
std::string checksum(const cpp_knn::PredictionResult& result) {
    std::uint64_t hash = 14695981039346656037ULL;
    const auto add = [&](unsigned char byte) { hash ^= byte; hash *= 1099511628211ULL; };
    for (std::size_t row = 0; row < result.labels.size(); ++row) {
        const auto* bytes = reinterpret_cast<const unsigned char*>(&result.scores[row]);
        for (std::size_t byte = 0; byte < sizeof(double); ++byte) add(bytes[byte]);
        add(static_cast<unsigned char>(result.labels[row]));
    }
    std::ostringstream output;
    output << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << hash;
    return output.str();
}

void configuration(std::ostream& output, const Arguments& args,
                   const cpp_knn::ExactKnnClassifier& classifier, std::size_t queries) {
    output << "\"model_id\":" << quoted(args.model_id)
           << ",\"config_id\":" << quoted(args.config_id)
           << ",\"weights\":" << quoted(std::string(cpp_knn::weight_mode_name(args.weights)))
           << ",\"threshold\":";
    if (args.threshold) output << *args.threshold; else output << "null";
    output << ",\"decision_rule\":" << quoted(args.threshold ? "score >= threshold" : "argmax; ties to class 0")
           << ",\"training_rows\":" << classifier.training_rows()
           << ",\"test_rows\":" << queries << ",\"feature_count\":" << classifier.feature_count()
           << ",\"k\":" << classifier.k() << ",\"batch_size\":" << args.batch_size
           << ",\"algorithm\":" << quoted(args.reference ? "scalar_reference" : "optimized")
           << ",\"selection\":" << quoted(std::string(cpp_knn::selection_method_name(args.selection))) << ',';
    identity(output);
}

void write_csv(const Arguments& args, const cpp_knn::PredictionResult& result) {
    if (args.predictions.empty()) return;
    if (!args.predictions.parent_path().empty()) std::filesystem::create_directories(args.predictions.parent_path());
    std::ofstream csv(args.predictions);
    if (!csv) throw std::runtime_error("cannot write prediction CSV");
    csv << "test_position,score_class_1,y_pred\n" << std::setprecision(17);
    for (std::size_t row = 0; row < result.labels.size(); ++row)
        csv << row << ',' << result.scores[row] << ',' << result.labels[row] << '\n';
    if (!csv) throw std::runtime_error("failed to write prediction CSV");
}

void write_result(const Arguments& args, const cpp_knn::ExactKnnClassifier& classifier,
                  const cpp_knn::PredictionResult& result, const std::vector<int>& labels,
                  const std::vector<double>& times, double fit_seconds) {
    if (!args.output.parent_path().empty()) std::filesystem::create_directories(args.output.parent_path());
    std::ofstream output(args.output);
    if (!output) throw std::runtime_error("cannot write output JSON");
    output << std::setprecision(17) << '{';
    configuration(output, args, classifier, result.labels.size());
    output << ",\"fit_seconds\":" << fit_seconds << ",\"warmup_count\":" << args.warmups
           << ",\"measured_run_count\":" << times.size() << ",\"prediction_seconds\":";
    array(output, times);
    output << ",\"median_prediction_seconds\":" << percentile(times, .5)
           << ",\"iqr_prediction_seconds\":" << percentile(times, .75) - percentile(times, .25)
           << ",\"predicted_labels\":"; array(output, result.labels);
    output << ",\"scores_class_1\":"; array(output, result.scores);
    output << ",\"positive_vote_counts\":"; array(output, result.positive_vote_counts);
    output << ",\"prediction_hash\":" << quoted(checksum(result))
           << ",\"timing_scope\":\"query validation, fresh exact search, voting, score and label allocation; excludes load, fit, startup, IPC, checksum and serialization\"";
    if (!labels.empty()) {
        std::size_t counts[4]{};
        for (std::size_t row = 0; row < labels.size(); ++row)
            ++counts[labels[row] * 2 + result.labels[row]];
        output << ",\"confusion_matrix\":{\"TN\":" << counts[0] << ",\"FP\":" << counts[1]
               << ",\"FN\":" << counts[2] << ",\"TP\":" << counts[3] << '}';
    }
    output << "}\n";
    if (!output) throw std::runtime_error("failed to write output JSON");
    write_csv(args, result);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--identity") {
            std::cout << '{'; identity(std::cout); std::cout << "}\n"; return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Exact single-thread C++20 KNN: --train-features CSV --train-labels CSV --test-features CSV --k N --output JSON [--predictions CSV] [--weights uniform|distance] [--threshold T] [--batch-size 32] [--algorithm optimized|reference] [--selection heap|nth] [--warmups N] [--runs N] [--model-id ID] [--config-id HASH] [--persistent true]\n";
            return 0;
        }
        const Arguments args = parse_arguments(argc, argv);
        auto training = cpp_knn::load_feature_csv(args.train_features);
        auto labels = cpp_knn::load_label_csv(args.train_labels);
        const auto queries = cpp_knn::load_feature_csv(args.test_features);
        std::vector<int> test_labels;
        if (!args.test_labels.empty()) {
            test_labels = cpp_knn::load_label_csv(args.test_labels);
            if (test_labels.size() != queries.rows) throw std::invalid_argument("test labels differ in length");
        }
        const auto fit_start = Clock::now();
        const cpp_knn::ExactKnnClassifier classifier(std::move(training), std::move(labels), args.k, args.weights, args.threshold);
        const double fit_seconds = std::chrono::duration<double>(Clock::now() - fit_start).count();
        const auto predict = [&]() { return args.reference ? classifier.predict_reference(queries)
            : classifier.predict_optimized(queries, args.batch_size, args.selection); };
        std::vector<double> warmup_seconds;
        for (std::size_t warmup = 0; warmup < args.warmups; ++warmup) {
            const auto start = Clock::now();
            const auto result = predict();
            warmup_seconds.push_back(std::chrono::duration<double>(Clock::now() - start).count());
            (void)result;
        }
        if (args.persistent) {
            std::cout << std::setprecision(17) << "{\"status\":\"READY\",\"fit_seconds\":" << fit_seconds << ',';
            configuration(std::cout, args, classifier, queries.rows);
            std::cout << ",\"warmup_seconds\":"; array(std::cout, warmup_seconds);
            std::cout << "}\n" << std::flush;
            std::string command;
            while (std::getline(std::cin, command)) {
                if (command == "EXIT") { std::cout << "{\"status\":\"EXIT\"}\n" << std::flush; return 0; }
                if (command != "PREDICT") throw std::invalid_argument("persistent command must be PREDICT or EXIT");
                const auto start = Clock::now();
                const auto result = predict();
                const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
                std::cout << "{\"status\":\"PREDICTION\",\"seconds\":" << seconds
                          << ",\"prediction_count\":" << result.labels.size()
                          << ",\"checksum\":" << quoted(checksum(result)) << "}\n" << std::flush;
            }
            return 0;
        }
        std::vector<double> seconds;
        cpp_knn::PredictionResult final_result;
        std::string prior_checksum;
        for (std::size_t run = 0; run < args.runs; ++run) {
            const auto start = Clock::now();
            auto result = predict();
            seconds.push_back(std::chrono::duration<double>(Clock::now() - start).count());
            const auto current_checksum = checksum(result);
            if (run && current_checksum != prior_checksum) throw std::runtime_error("predictions changed between runs");
            prior_checksum = current_checksum;
            final_result = std::move(result);
        }
        write_result(args, classifier, final_result, test_labels, seconds, fit_seconds);
        std::cerr << "C++ " << args.model_id << ": " << final_result.labels.size() << " rows -> " << args.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cpp_knn error: " << error.what() << '\n';
        return 1;
    }
}
