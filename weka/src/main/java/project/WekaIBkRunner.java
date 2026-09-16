package project;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.EuclideanDistance;
import weka.core.ManhattanDistance;
import weka.core.NormalizableDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Utils;
import weka.core.Version;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.neighboursearch.LinearNNSearch;

import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Genuine Weka IBk over the common numeric features, with no independent split. */
public final class WekaIBkRunner {
    private static final Gson JSON = new GsonBuilder().setPrettyPrinting().create();

    private static JsonObject readJson(Path path) throws Exception {
        return JSON.fromJson(Files.readString(path, StandardCharsets.UTF_8), JsonObject.class);
    }

    private static void write(Path path, String text) throws Exception {
        if (path.toAbsolutePath().getParent() != null) Files.createDirectories(path.toAbsolutePath().getParent());
        Files.writeString(path, text, StandardCharsets.UTF_8);
    }

    private static void validateData(Instances data) {
        data.setClassIndex(data.numAttributes() - 1);
        if (!data.classAttribute().isNominal() || data.numClasses() != 2
                || data.classAttribute().indexOfValue("0") < 0 || data.classAttribute().indexOfValue("1") < 0)
            throw new IllegalArgumentException("The final class attribute must be nominal {0,1}");
        if (data.numInstances() == 0) throw new IllegalArgumentException("Empty ARFF dataset");
        for (int j = 0; j < data.classIndex(); j++) {
            if (!data.attribute(j).isNumeric()) throw new IllegalArgumentException("All predictors must be numeric");
            String name = data.attribute(j).name();
            if (name.equalsIgnoreCase("ID") || name.equalsIgnoreCase("target"))
                throw new IllegalArgumentException("ID/target cannot be predictive features");
        }
        for (Instance instance : data) {
            for (int j = 0; j < data.numAttributes(); j++)
                if (!Double.isFinite(instance.value(j))) throw new IllegalArgumentException("ARFF contains missing/nonfinite values");
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 0 || List.of(args).contains("--help")) {
            System.out.println("Weka " + Version.VERSION + " genuine IBk runner\n"
                    + "java -jar weka/target/knn-weka-runner.jar --train <train.arff> --test <test.arff> "
                    + "--k <selected-k> --predictions <predictions.csv> [--metrics <evaluation.txt>] "
                    + "[--selected-parameters <selection.json>] [--ids <test_ids.csv>] "
                    + "[--metric euclidean|manhattan] [--weights uniform|distance]\n"
                    + "Requires test row IDs from python -m src.preprocessing. "
                    + "Defaults preserve the accepted no-CV, uniform-vote, "
                    + "EuclideanDistance dontNormalize=true behavior.");
            return;
        }
        Path localWekaHome = Path.of("weka", "target", "weka-home").toAbsolutePath();
        Files.createDirectories(localWekaHome);
        weka.core.Environment.getSystemWide().addVariable("WEKA_HOME", localWekaHome.toString());
        Set<String> allowed = Set.of("--train", "--test", "--k", "--predictions", "--metrics", "--ids", "--selected-parameters", "--metric", "--weights");
        Map<String, String> options = new HashMap<>();
        for (int i = 0; i < args.length; i += 2) {
            if (i + 1 >= args.length || !allowed.contains(args[i]) || options.put(args[i], args[i + 1]) != null)
                throw new IllegalArgumentException("Invalid, duplicate or incomplete argument: " + args[i]);
        }
        for (String key : List.of("--train", "--test", "--k", "--predictions"))
            if (!options.containsKey(key)) throw new IllegalArgumentException("Required argument: " + key);
        Path trainPath = Path.of(options.get("--train")).toAbsolutePath();
        Path testPath = Path.of(options.get("--test")).toAbsolutePath();
        Path predictionPath = Path.of(options.get("--predictions")).toAbsolutePath();
        Path idsPath = Path.of(options.getOrDefault("--ids", testPath.resolveSibling("test_ids.csv").toString()));
        int k = Integer.parseInt(options.get("--k"));
        if (k < 1) throw new IllegalArgumentException("k must be >= 1");
        String metric = options.getOrDefault("--metric", "euclidean");
        String weights = options.getOrDefault("--weights", "uniform");
        if (!Set.of("euclidean", "manhattan").contains(metric))
            throw new IllegalArgumentException("metric must be euclidean or manhattan");
        if (!Set.of("uniform", "distance").contains(weights))
            throw new IllegalArgumentException("weights must be uniform or distance");
        Path selectionPath = options.containsKey("--selected-parameters") ? Path.of(options.get("--selected-parameters")) : null;
        if (selectionPath != null) {
            JsonObject selection = readJson(selectionPath);
            if (selection.get("selected_k").getAsInt() != k)
                throw new IllegalArgumentException("Selected k mismatch");
            if (selection.has("metric") && !selection.get("metric").getAsString().equals(metric))
                throw new IllegalArgumentException("Selected metric mismatch");
            if (selection.has("weights") && !selection.get("weights").getAsString().equals(weights))
                throw new IllegalArgumentException("Selected weights mismatch");
        }
        Instances train = DataSource.read(trainPath.toString());
        Instances test = DataSource.read(testPath.toString());
        validateData(train);
        validateData(test);
        if (!train.equalHeaders(test)) throw new IllegalArgumentException(train.equalHeadersMsg(test));
        if (k > train.numInstances()) throw new IllegalArgumentException("k exceeds training rows");
        List<CSVRecord> ids;
        try (Reader reader = Files.newBufferedReader(idsPath, StandardCharsets.UTF_8);
             CSVParser csv = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).build().parse(reader)) {
            ids = csv.getRecords();
        }
        if (ids.size() != test.numInstances()) throw new IllegalArgumentException("Test IDs have incorrect row count");
        Set<String> seen = new HashSet<>();
        for (CSVRecord row : ids) if (!seen.add(row.get("row_id"))) throw new IllegalArgumentException("Duplicate test row ID");

        NormalizableDistance distance = metric.equals("euclidean")
                ? new EuclideanDistance() : new ManhattanDistance();
        distance.setDontNormalize(true);
        LinearNNSearch search = new LinearNNSearch();
        search.setSkipIdentical(false);
        search.setDistanceFunction(distance);
        IBk classifier = new IBk(k);
        classifier.setCrossValidate(false);
        int weightingMode = weights.equals("uniform") ? IBk.WEIGHT_NONE : IBk.WEIGHT_INVERSE;
        classifier.setDistanceWeighting(new SelectedTag(weightingMode, IBk.TAGS_WEIGHTING));
        classifier.setNearestNeighbourSearchAlgorithm(search);
        long start = System.nanoTime();
        classifier.buildClassifier(train);
        double fitSeconds = (System.nanoTime() - start) / 1e9;
        int positiveIndex = train.classAttribute().indexOfValue("1");
        double[][] distributions = new double[test.numInstances()][];
        int[] predictedIndices = new int[test.numInstances()];
        start = System.nanoTime();
        for (int i = 0; i < test.numInstances(); i++) {
            Instance query = (Instance) test.instance(i).copy();
            query.setDataset(test);
            query.setClassMissing();
            distributions[i] = classifier.distributionForInstance(query);
            predictedIndices[i] = Utils.maxIndex(distributions[i]);
        }
        double predictionSeconds = (System.nanoTime() - start) / 1e9;

        Evaluation evaluation = new Evaluation(train);
        Files.createDirectories(predictionPath.getParent());
        try (CSVPrinter csv = new CSVPrinter(Files.newBufferedWriter(predictionPath, StandardCharsets.UTF_8),
                CSVFormat.DEFAULT.builder().setHeader("test_index", "row_id", "original_id", "y_true", "y_pred",
                        "probability_class_1", "selected_k").build())) {
            for (int i = 0; i < test.numInstances(); i++) {
                csv.printRecord(i, ids.get(i).get("row_id"), ids.get(i).get("original_id"),
                        test.classAttribute().value((int) test.instance(i).classValue()),
                        train.classAttribute().value(predictedIndices[i]), distributions[i][positiveIndex],
                        k);
                evaluation.evaluateModelOnceAndRecordPrediction(distributions[i], test.instance(i));
            }
        }
        Map<String, Object> runtime = new java.util.LinkedHashMap<>();
        runtime.put("implementation", "weka");
        runtime.put("selected_k", k);
        runtime.put("fit_seconds", fitSeconds);
        runtime.put("prediction_seconds", predictionSeconds);
        runtime.put("milliseconds_per_sample", 1000 * predictionSeconds / test.numInstances());
        runtime.put("weka_version", Version.VERSION);
        runtime.put("java_version", System.getProperty("java.version"));
        runtime.put("distance_normalization", !distance.getDontNormalize());
        runtime.put("internal_cross_validation", classifier.getCrossValidate());
        runtime.put("metric", metric);
        runtime.put("distance_weighting", weights.equals("uniform") ? "none" : "inverse");
        runtime.put("search", "LinearNNSearch");
        runtime.put("classifier_options", Utils.joinOptions(classifier.getOptions()));
        runtime.put("timing_scope", "one buildClassifier; one distributionForInstance loop plus argmax; I/O/evaluation/JVM startup excluded; no warmup");
        runtime.put("probability_semantics", "Unmodified Weka IBk nominal distribution; includes native smoothing and boundary distance ties");
        String stem = predictionPath.getFileName().toString().replaceFirst("\\.csv$", "");
        write(predictionPath.resolveSibling(stem + ".runtime.json"), JSON.toJson(runtime) + "\n");
        Path evaluationPath = Path.of(options.getOrDefault("--metrics", predictionPath.resolveSibling("weka_evaluation.txt").toString()));
        write(evaluationPath, "Weka " + Version.VERSION + "\n" + JSON.toJson(runtime) + "\n"
                + evaluation.toSummaryString("Native Weka Evaluation\n", false)
                + evaluation.toClassDetailsString() + evaluation.toMatrixString());
        System.out.printf(java.util.Locale.ROOT, "Weka %s IBk: k=%d, metric=%s, weights=%s, %d predictions, %.6fs; dontNormalize=%s -> %s%n",
                Version.VERSION, k, metric, weights, test.numInstances(), predictionSeconds, distance.getDontNormalize(), predictionPath);
    }
}
