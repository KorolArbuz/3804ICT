from src.common.model_runner import runner_parser

from .runner import run


def main(argv=None):
    parser = runner_parser("Run scikit-learn KNN on the processed arrays.", "sklearn")
    args = parser.parse_args(argv)
    run(
        args.train,
        args.test,
        args.output,
        args.k,
        args.selected_parameters,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
