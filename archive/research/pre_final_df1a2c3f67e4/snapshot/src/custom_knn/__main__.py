from src.common.config import DEFAULT_BATCH_SIZE
from src.common.model_runner import runner_parser

from .runner import run


def main(argv=None):
    parser = runner_parser("Run the manually implemented KNN classifier.", "custom")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args(argv)
    run(
        args.train,
        args.test,
        args.output,
        args.k,
        args.batch_size,
        args.selected_parameters,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
