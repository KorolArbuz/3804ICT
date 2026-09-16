"""Re-evaluate current-run saved predictions without model inference."""
import argparse
from pathlib import Path
from src.common.experiment import load_configs
from src.common.utils import read_json
from .final import evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = read_json(args.run_dir / "run_manifest.json")
    configs = load_configs()
    if manifest["config_hashes"] != {k: v["config_hash"] for k, v in configs.items()}:
        raise ValueError("Run configuration differs from current frozen configurations")
    table, _, _ = evaluate(
        args.run_dir,
        manifest["run_id"],
        manifest["completed_implementations"],
        configs,
        read_json(args.run_dir / "data_manifest.json"),
    )
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
