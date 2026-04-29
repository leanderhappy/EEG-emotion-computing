from __future__ import annotations

import argparse
from pathlib import Path

from deap_task2.dataset import export_training_dataset
from deap_task2.io import resolve_default_metadata_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build train-ready DEAP task2 bundles")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root that contains task1 outputs",
    )
    parser.add_argument(
        "--metadata-source",
        type=Path,
        default=None,
        help="Path to metadata_csv.zip or extracted metadata folder",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=None,
        help="Custom task1 processed root, default is <project-root>/data/processed",
    )
    parser.add_argument(
        "--interim-root",
        type=Path,
        default=None,
        help="Custom task1 interim root, default is <project-root>/data/interim",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Task2 output root, default is <project-root>/data/task2",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject ids like s01 s02",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Scores strictly above this value are marked as high",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    metadata_source = args.metadata_source or resolve_default_metadata_source(args.project_root)
    results, manifest, summary = export_training_dataset(
        project_root=args.project_root,
        metadata_source=metadata_source,
        processed_root=args.processed_root,
        interim_root=args.interim_root,
        output_root=args.output_root,
        subject_ids=args.subjects,
        threshold=args.threshold,
    )
    print(
        "Built task2 dataset "
        f"for {len(results)} subjects and {len(manifest)} trials. "
        f"Summary saved with {len(summary)} rows."
    )


if __name__ == "__main__":
    main()
