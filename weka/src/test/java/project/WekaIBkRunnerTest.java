package project;

import org.junit.Test;
import static org.junit.Assert.*;
import java.io.StringReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;
import com.google.gson.JsonParser;
import weka.classifiers.lazy.IBk;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;

public class WekaIBkRunnerTest {
    private Instances data(String rows) throws Exception {
        Instances result = new Instances(new StringReader("@relation test\n@attribute x numeric\n@attribute class {0,1}\n@data\n" + rows));
        result.setClassIndex(1);
        return result;
    }

    @Test public void genuineSettingsAndDistributionAreRetained() throws Exception {
        Instances train = data("0,0\n1,1\n3,1\n");
        Instances query = data("0.25,?\n");
        IBk model = WekaIBkRunner.createClassifier(2, train);
        assertEquals(IBk.class, model.getClass());
        assertFalse(model.getCrossValidate());
        assertEquals(IBk.WEIGHT_INVERSE, model.getDistanceWeighting().getSelectedTag().getID());
        LinearNNSearch search = (LinearNNSearch) model.getNearestNeighbourSearchAlgorithm();
        assertFalse(search.getSkipIdentical());
        assertTrue(((EuclideanDistance) search.getDistanceFunction()).getDontNormalize());
        double nativeScore = model.distributionForInstance(query.instance(0))[1];
        WekaIBkRunner.Prediction prediction = WekaIBkRunner.predict(model, query, 0.3315411365543412);
        assertEquals(nativeScore, prediction.scores[0], 0);
        assertEquals(nativeScore >= 0.3315411365543412 ? 1 : 0, prediction.labels[0]);
    }

    @Test public void thresholdEqualityIsPositive() {
        double threshold = WekaIBkRunner.FINAL_THRESHOLD;
        assertEquals(1, WekaIBkRunner.thresholdLabel(threshold, threshold));
        assertEquals(0, WekaIBkRunner.thresholdLabel(Math.nextDown(threshold), threshold));
    }

    @Test public void duplicateAndBoundaryScoresStayNative() throws Exception {
        Instances train = data("0,0\n0,1\n-1,1\n1,0\n");
        Instances query = data("0,?\n2,?\n");
        IBk model = WekaIBkRunner.createClassifier(1, train);
        WekaIBkRunner.Prediction first = WekaIBkRunner.predict(model, query, 0.5);
        WekaIBkRunner.Prediction second = WekaIBkRunner.predict(model, query, 0.5);
        assertEquals(0.5, first.scores[0], 1e-15);
        for (int i = 0; i < 2; i++) {
            assertEquals(model.distributionForInstance(query.instance(i))[1], first.scores[i], 0);
            assertTrue(Double.isFinite(first.scores[i]));
        }
        assertEquals(first.checksum(), second.checksum());
        assertNotSame(first.scores, second.scores);
        assertNotSame(first.labels, second.labels);
    }

    @Test public void queryMissingClassAndTrainingLabelsEnforced() throws Exception {
        WekaIBkRunner.validateData(data("1,?\n"), true);
        assertThrows(IllegalArgumentException.class, () -> WekaIBkRunner.validateData(data("1,0\n"), true));
        assertThrows(IllegalArgumentException.class, () -> WekaIBkRunner.validateData(data("1,?\n"), false));
        assertThrows(IllegalArgumentException.class, () -> WekaIBkRunner.validateData(data("?,?\n"), true));
    }

    @Test public void rejectsWrongClassOrderAndInvalidK() throws Exception {
        Instances reversed = new Instances(new StringReader("@relation t\n@attribute x numeric\n@attribute class {1,0}\n@data\n1,?\n"));
        assertThrows(IllegalArgumentException.class, () -> WekaIBkRunner.validateData(reversed, true));
        Instances train = data("0,0\n1,1\n");
        assertThrows(IllegalArgumentException.class, () -> WekaIBkRunner.createClassifier(0, train));
        assertThrows(IllegalArgumentException.class, () -> WekaIBkRunner.createClassifier(3, train));
        assertThrows(IllegalArgumentException.class, () -> WekaIBkRunner.thresholdLabel(Double.NaN, 0.5));
    }

    @Test public void csvContractAndPathsWithSpaces() throws Exception {
        Path dir = Files.createTempDirectory("weka test ");
        Path output = dir.resolve("predictions with spaces.csv");
        try {
            WekaIBkRunner.writePredictions(output, new WekaIBkRunner.Prediction(new double[]{0.1,0.9}, new int[]{0,1}, 0));
            List<String> lines = Files.readAllLines(output);
            assertEquals("test_position,score_class_1,y_pred", lines.get(0));
            assertEquals("1,0.9,1", lines.get(2));
        } finally { Files.deleteIfExists(output); Files.deleteIfExists(dir); }
    }

    @Test public void persistentWorkerMaterializesEveryTrialInSameJvm() throws Exception {
        Path dir = Files.createTempDirectory("weka worker test ");
        Path train = dir.resolve("train data.arff");
        Path test = dir.resolve("test data.arff");
        Path log = dir.resolve("stderr.log");
        String header = "@relation t\n@attribute x numeric\n@attribute class {0,1}\n@data\n";
        Files.writeString(train, header + "0,0\n1,1\n3,1\n");
        Files.writeString(test, header + "0.25,?\n2,?\n");
        String javaExecutable = Path.of(System.getProperty("java.home"), "bin", "java").toString();
        Process process = new ProcessBuilder(javaExecutable, "-cp", System.getProperty("java.class.path"),
                "project.WekaIBkRunner", "--train", train.toString(), "--test", test.toString(),
                "--k", "2", "--worker", "--warmups", "1").redirectError(log.toFile()).start();
        try {
            process.getOutputStream().write("PREDICT\nPREDICT\nEXIT\n".getBytes(java.nio.charset.StandardCharsets.UTF_8));
            process.getOutputStream().close();
            assertTrue("Worker timeout", process.waitFor(30, TimeUnit.SECONDS));
            assertEquals(new String(Files.readAllBytes(log), java.nio.charset.StandardCharsets.UTF_8), 0, process.exitValue());
            List<String> lines = new String(process.getInputStream().readAllBytes(), java.nio.charset.StandardCharsets.UTF_8).lines().collect(java.util.stream.Collectors.toList());
            assertEquals(4, lines.size());
            assertEquals("READY", JsonParser.parseString(lines.get(0)).getAsJsonObject().get("status").getAsString());
            String checksum = null;
            for (int i = 1; i <= 2; i++) {
                com.google.gson.JsonObject result = JsonParser.parseString(lines.get(i)).getAsJsonObject();
                assertEquals("PREDICT", result.get("status").getAsString());
                assertTrue(result.get("seconds").getAsDouble() > 0);
                assertEquals(2, result.get("n_query").getAsInt());
                if (checksum != null) assertEquals(checksum, result.get("checksum").getAsString());
                checksum = result.get("checksum").getAsString();
            }
            assertEquals("EXIT", JsonParser.parseString(lines.get(3)).getAsJsonObject().get("status").getAsString());
        } finally {
            process.destroyForcibly();
            process.waitFor(10, TimeUnit.SECONDS);
            Files.deleteIfExists(train); Files.deleteIfExists(test); Files.deleteIfExists(log); Files.deleteIfExists(dir);
        }
    }
}
