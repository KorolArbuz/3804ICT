from pathlib import Path
import shutil
import subprocess

from src.common.config import PROJECT_ROOT, WEKA_DIR
from src.common.utils import read_json

JAR = WEKA_DIR / "target/knn-weka-runner.jar"


def find_maven():
    return shutil.which("mvn")


def build(force=False):
    if not force and JAR.exists():
        return JAR
    maven = find_maven()
    if maven is None:
        raise FileNotFoundError("Maven is unavailable; install Maven and ensure mvn is on PATH")

    command = [maven, "-B", "-ntp", "-f", str(WEKA_DIR / "pom.xml"), "clean", "package"]
    print("Building genuine Weka runner with Maven", flush=True)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        errors="replace",
    )

    log = WEKA_DIR / "target/maven_build.log"
    log.parent.mkdir(exist_ok=True)
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")

    if completed.returncode or not JAR.exists():
        raise RuntimeError(f"Maven build failed ({completed.returncode}); see {log}")

    print(f"Maven BUILD SUCCESS -> {JAR}", flush=True)
    return JAR


def run(train, test, k, predictions, selected_parameters=None, force_rebuild=False):
    jar = build(force_rebuild)
    java = shutil.which("java")
    if java is None:
        raise FileNotFoundError("Java is unavailable; install a JDK (11 or later)")

    command = [
        java,
        "-XX:ActiveProcessorCount=1",
        "-jar",
        str(jar),
        "--train",
        str(train),
        "--test",
        str(test),
        "--k",
        str(k),
        "--predictions",
        str(predictions),
    ]
    if selected_parameters is not None:
        command.extend(["--selected-parameters", str(selected_parameters)])

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        errors="replace",
    )

    log = Path(predictions).parent / "weka_execution.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    print(completed.stdout, end="", flush=True)

    if completed.returncode:
        raise RuntimeError(f"Java/Weka execution failed ({completed.returncode}); see {log}")

    return read_json(Path(predictions).with_suffix(".runtime.json"))
