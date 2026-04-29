from __future__ import annotations

import argparse
from pathlib import Path

from deap_task2.io import build_task2_layout, resolve_default_metadata_source
from deap_task2.labels import export_task2_labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export DEAP task2 trial labels")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root that contains data/processed and data/interim",
    )
    parser.add_argument(
        "--metadata-source",
        type=Path,
        default=None,
        help="Path to metadata_csv.zip or extracted metadata folder",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Custom output CSV path",
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
    layout = build_task2_layout(args.project_root)
    output_path = args.output_path or (layout["labels"] / "deap_trial_labels.csv")
    labels = export_task2_labels(
        metadata_source=metadata_source,
        output_path=output_path,
        threshold=args.threshold,
    )
    print(f"Saved {len(labels)} task2 labels to {output_path}")


if __name__ == "__main__":
    main()
