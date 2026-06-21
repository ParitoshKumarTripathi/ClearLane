from __future__ import annotations

import argparse
import json
from pathlib import Path

from clearlane.config import load_config, project_root
from clearlane.pipeline import run_pipeline
from clearlane.preprocessing import load_and_clean_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clearlane",
        description="ClearLane reproducible ML and deployment pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pipeline = subparsers.add_parser("pipeline", help="Run preprocessing, training, evaluation, and export")
    pipeline.add_argument("--input", required=True, help="Path to the competition CSV")
    pipeline.add_argument("--config", default=None, help="Path to config.yaml")
    pipeline.add_argument("--root", default=str(project_root()), help="Project output directory")
    pipeline.add_argument(
        "--skip-intermediate",
        action="store_true",
        help="Do not save large cleaned and feature parquet files",
    )

    inspect = subparsers.add_parser("inspect", help="Validate and summarize the raw CSV")
    inspect.add_argument("--input", required=True)
    inspect.add_argument("--config", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "pipeline":
        result = run_pipeline(
            input_path=args.input,
            config_path=args.config,
            root=Path(args.root),
            save_intermediate=not args.skip_intermediate,
        )
        print(json.dumps(result["data_summary"], indent=2))
    elif args.command == "inspect":
        config = load_config(args.config)
        _, summary = load_and_clean_records(args.input, config)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
