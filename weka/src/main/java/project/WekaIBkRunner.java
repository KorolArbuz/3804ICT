package project;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import weka.classifiers.lazy.IBk;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.neighboursearch.LinearNNSearch;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;

/** Genuine Weka scores on the shared prepared matrix; thresholding is external to IBk. */
public final class WekaIBkRunner {
    static final double FINAL_THRESHOLD = 0.3315411365543412;
    private static final Gson JSON = new Gson();
    private static final Gson PRETTY_JSON = new GsonBuilder().setPrettyPrinting().create();

    static void validateData(Instances data, boolean query) {
        if (data == null || data.numAttributes() < 2 || data.numInstances() == 0)
            throw new IllegalArgumentException("ARFF must contain predictors and at least one row");
        data.setClassIndex(data.numAttributes() - 1);
        if (!data.classAttribute().isNominal() || data.numClasses() != 2
                || !data.classAttribute().value(0).equals("0") || !data.classAttribute().value(1).equals("1"))
            throw new IllegalArgumentException("The final class attribute must be nominal {0,1}, in that order");
        for (int j = 0; j < data.classIndex(); j++) {
            if (!data.attribute(j).isNumeric()) throw new IllegalArgumentException("All predictors must be numeric");
            if (Set.of("id", "target", "row_id", "original_id", "y_true", "default.payment.next.month")
                    .contains(data.attribute(j).name().toLowerCase(Locale.ROOT)))
                throw new IllegalArgumentException("ID/target cannot be predictive features");
        }
        for (Instance row : data) {
            for (int j = 0; j < data.classIndex(); j++)
                if (!Double.isFinite(row.value(j))) throw new IllegalArgumentException("Nonfinite predictor");
            if (query != row.classIsMissing())
                throw new IllegalArgumentException(query ? "Query class must be missing" : "Training class must be present");
        }
    }

    static IBk createClassifier(int k, Instances train) throws Exception {
        if (k < 1 || k > train.numInstances()) throw new IllegalArgumentException("k outside training row count");
        EuclideanDistance distance = new EuclideanDistance();
        distance.setDontNormalize(true);
        LinearNNSearch search = new LinearNNSearch();
        search.setSkipIdentical(false);
        search.setDistanceFunction(distance);
        IBk classifier = new IBk(k);
        classifier.setCrossValidate(false);
        classifier.setDistanceWeighting(new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING));
        classifier.setNearestNeighbourSearchAlgorithm(search);
        classifier.buildClassifier(train);
        return classifier;
    }

    static int thresholdLabel(double score, double threshold) {
        if (!Double.isFinite(score) || score < 0 || score > 1)
            throw new IllegalArgumentException("Invalid class-1 score");
        if (!Double.isFinite(threshold) || threshold < 0 || threshold > 1)
            throw new IllegalArgumentException("Invalid threshold");
        return score >= threshold ? 1 : 0;
    }

    static final class Prediction {
        final double[] scores;
        final int[] labels;
        final double seconds;
        Prediction(double[] scores, int[] labels, double seconds) {
            this.scores = scores; this.labels = labels; this.seconds = seconds;
        }
        String checksum() throws Exception {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            ByteBuffer buffer = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < scores.length; i++) {
                buffer.clear(); buffer.putDouble(scores[i]); buffer.putInt(labels[i]);
                digest.update(buffer.array());
            }
            StringBuilder hex = new StringBuilder();
            for (byte value : digest.digest()) hex.append(String.format("%02x", value & 0xff));
            return hex.toString();
        }
    }

    static Prediction predict(IBk classifier, Instances test, double threshold) throws Exception {
        long start = System.nanoTime();
        // Validation, allocation, full neighbour searches and labels are inside the timer.
        validateData(test, true);
        double[] scores = new double[test.numInstances()];
        int[] labels = new int[test.numInstances()];
        int positiveIndex = test.classAttribute().indexOfValue("1");
        for (int i = 0; i < test.numInstances(); i++) {
            Instance query = (Instance) test.instance(i).copy();
            query.setDataset(test); query.setClassMissing();
            double[] distribution = classifier.distributionForInstance(query);
            if (distribution.length != 2 || !Double.isFinite(distribution[0])
                    || Math.abs(distribution[0] + distribution[1] - 1.0) > 1e-12)
                throw new IllegalStateException("Invalid Weka distribution");
            scores[i] = distribution[positiveIndex];
            labels[i] = thresholdLabel(scores[i], threshold);
        }
        return new Prediction(scores, labels, (System.nanoTime() - start) / 1e9);
    }

    static Map<String, Object> metadata(IBk classifier, Instances train, Instances test,
                                         double threshold, double fitSeconds) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("implementation", "final_weka"); result.put("k", classifier.getKNN());
        result.put("threshold", threshold); result.put("decision_rule", "score_class_1 >= threshold");
        result.put("fit_seconds", fitSeconds); result.put("n_train", train.numInstances());
        result.put("n_query", test.numInstances()); result.put("d", train.numAttributes() - 1);
        result.put("weka_version", Version.VERSION); result.put("java_version", System.getProperty("java.version"));
        result.put("distance_normalization", false); result.put("skip_identical", false);
        result.put("internal_cross_validation", false); result.put("metric", "euclidean");
        result.put("distance_weighting", "inverse"); result.put("search", "LinearNNSearch");
        result.put("classifier_options", Utils.joinOptions(classifier.getOptions()));
        result.put("compute_threads", 1);
        result.put("thread_note", "Sequential IBk queries; JVM may use service/JIT/GC threads");
        result.put("timing_scope", "Ready model and prepared query Instances -> validation, allocation, genuine distributions, threshold labels; excludes load, fit, JVM startup, IPC, checksum and file output");
        result.put("score_semantics", "Unmodified Weka IBk distribution: native inverse weights, smoothing and boundary ties; not calibrated to custom scores");
        return result;
    }

    static void writePredictions(Path path, Prediction prediction) throws Exception {
        Files.createDirectories(path.toAbsolutePath().getParent());
        try (CSVPrinter csv = new CSVPrinter(Files.newBufferedWriter(path, StandardCharsets.UTF_8),
                CSVFormat.DEFAULT.builder().setHeader("test_position", "score_class_1", "y_pred").build())) {
            for (int i = 0; i < prediction.scores.length; i++)
                csv.printRecord(i, prediction.scores[i], prediction.labels[i]);
        }
    }

    private static void worker(PrintStream protocol, IBk classifier, Instances train, Instances test,
                               double threshold, int warmups, double fitSeconds) throws Exception {
        List<Map<String, Object>> warmupResults = new ArrayList<>();
        for (int i = 0; i < warmups; i++) {
            Prediction prediction = predict(classifier, test, threshold);
            warmupResults.add(Map.of("warmup", i, "seconds", prediction.seconds, "checksum", prediction.checksum()));
        }
        Map<String, Object> ready = metadata(classifier, train, test, threshold, fitSeconds);
        ready.put("status", "READY"); ready.put("warmups", warmupResults);
        protocol.println(JSON.toJson(ready)); protocol.flush();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8))) {
            String command;
            while ((command = reader.readLine()) != null) {
                if (command.equals("EXIT")) { protocol.println("{\"status\":\"EXIT\"}"); protocol.flush(); return; }
                if (!command.equals("PREDICT")) throw new IllegalArgumentException("Unknown worker command: " + command);
                Prediction prediction = predict(classifier, test, threshold);
                protocol.println(JSON.toJson(Map.of("status", "PREDICT", "seconds", prediction.seconds,
                        "checksum", prediction.checksum(), "n_query", prediction.scores.length)));
                protocol.flush();
            }
        }
    }

    public static void main(String[] args) throws Exception {
        // Reserve stdout even if a Weka dependency logs to System.out.
        PrintStream protocol = System.out;
        System.setOut(System.err);
        if (args.length == 0 || List.of(args).contains("--help")) {
            protocol.println("Genuine Weka 3.8.6 final IBk: --train train.arff --test test.arff "
                    + "[--k 101] [--threshold 0.3315411365543412] --predictions output.csv "
                    + "OR --worker [--warmups 3]. Query class must be missing; class order is {0,1}.");
            return;
        }
        Map<String, String> options = new HashMap<>();
        Set<String> allowed = Set.of("--train", "--test", "--k", "--threshold", "--predictions", "--warmups");
        boolean workerMode = false;
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--worker") && !workerMode) { workerMode = true; continue; }
            if (!allowed.contains(args[i]) || i + 1 >= args.length || options.containsKey(args[i]))
                throw new IllegalArgumentException("Invalid, duplicate or incomplete argument: " + args[i]);
            String key = args[i]; options.put(key, args[++i]);
        }
        for (String key : List.of("--train", "--test"))
            if (!options.containsKey(key)) throw new IllegalArgumentException("Required argument: " + key);
        if (!workerMode && !options.containsKey("--predictions")) throw new IllegalArgumentException("Required argument: --predictions");
        int k = Integer.parseInt(options.getOrDefault("--k", "101"));
        double threshold = Double.parseDouble(options.getOrDefault("--threshold", Double.toString(FINAL_THRESHOLD)));
        thresholdLabel(0, threshold);
        int warmups = Integer.parseInt(options.getOrDefault("--warmups", "3"));
        if (warmups < 0) throw new IllegalArgumentException("Negative warmups");
        Path localWekaHome = Path.of("weka", "target", "weka-home").toAbsolutePath();
        Files.createDirectories(localWekaHome);
        weka.core.Environment.getSystemWide().addVariable("WEKA_HOME", localWekaHome.toString());
        Instances train = DataSource.read(Path.of(options.get("--train")).toAbsolutePath().toString());
        Instances test = DataSource.read(Path.of(options.get("--test")).toAbsolutePath().toString());
        validateData(train, false); validateData(test, true);
        if (!train.equalHeaders(test)) throw new IllegalArgumentException(train.equalHeadersMsg(test));
        long start = System.nanoTime();
        IBk classifier = createClassifier(k, train);
        double fitSeconds = (System.nanoTime() - start) / 1e9;
        if (workerMode) {
            worker(protocol, classifier, train, test, threshold, warmups, fitSeconds);
        } else {
            Prediction prediction = predict(classifier, test, threshold);
            Path path = Path.of(options.get("--predictions")).toAbsolutePath();
            writePredictions(path, prediction);
            Map<String, Object> runtime = metadata(classifier, train, test, threshold, fitSeconds);
            runtime.put("prediction_seconds", prediction.seconds); runtime.put("checksum", prediction.checksum());
            runtime.put("warmups", 0);
            String stem = path.getFileName().toString().replaceFirst("\\.csv$", "");
            Files.writeString(path.resolveSibling(stem + ".runtime.json"), PRETTY_JSON.toJson(runtime) + "\n", StandardCharsets.UTF_8);
            protocol.println(JSON.toJson(Map.of("status", "COMPLETE", "n_query", prediction.scores.length,
                    "checksum", prediction.checksum())));
        }
    }
}
