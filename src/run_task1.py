from __future__ import annotations

import argparse
from pathlib import Path

from deap_task1.io import list_subject_files, resolve_default_data_root, save_dataframe
from deap_task1.pipeline import result_to_frame, run_subject_pipeline


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("limit must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run DEAP task 1 preprocessing pipeline.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=resolve_default_data_root(project_root),
        help="Directory containing raw DEAP BDF files.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Optional subject id such as s01.",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Optional limit on number of subjects to process.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    bdf_files = list_subject_files(args.data_root, subject_id=args.subject)
    if args.limit is not None:
        bdf_files = bdf_files[: args.limit]

    results = [run_subject_pipeline(path, project_root=project_root) for path in bdf_files]
    save_dataframe(result_to_frame(results), project_root / "output" / "task1_run_summary.csv")


if __name__ == "__main__":
    main()
