from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd

from deap_task2.constants import (
    LABELS_DIRNAME,
    MANIFESTS_DIRNAME,
    NPZ_DIRNAME,
    TASK2_ROOT_DIRNAME,
)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_first_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find any existing path in candidates: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def resolve_default_metadata_source(project_root: Path) -> Path:
    return find_first_existing_path(
        [
            project_root / "data" / "raw" / "DEAP" / "metadata_csv.zip",
            project_root / "data" / "raw" / "DEAP" / "metadata_csv",
        ]
    )


def resolve_csv_member_name(source: Path, filename: str) -> str:
    if source.is_dir():
        return filename

    if source.suffix.lower() != ".zip":
        raise ValueError(f"Unsupported metadata source: {source}")

    with zipfile.ZipFile(source) as archive:
        members = archive.namelist()

    for member in members:
        if Path(member).name.lower() == filename.lower():
            return member

    raise FileNotFoundError(f"Could not find {filename} inside {source}")


def read_csv_from_source(source: Path, filename: str) -> pd.DataFrame:
    source = Path(source)
    if source.is_dir():
        target = source / filename
        if not target.exists():
            raise FileNotFoundError(f"Missing metadata file: {target}")
        return pd.read_csv(target)

    if source.suffix.lower() == ".zip":
        member_name = resolve_csv_member_name(source, filename)
        with zipfile.ZipFile(source) as archive:
            with archive.open(member_name) as handle:
                return pd.read_csv(handle)

    raise ValueError(f"Unsupported metadata source: {source}")


def build_task2_layout(project_root: Path, output_root: Path | None = None) -> dict[str, Path]:
    root = ensure_directory(output_root or (project_root / "data" / TASK2_ROOT_DIRNAME))
    labels = ensure_directory(root / LABELS_DIRNAME)
    npz = ensure_directory(root / NPZ_DIRNAME)
    manifests = ensure_directory(root / MANIFESTS_DIRNAME)
    return {
        "root": root,
        "labels": labels,
        "npz": npz,
        "manifests": manifests,
    }


def to_project_relative(path: Path, project_root: Path) -> str:
    try:
        relative = path.relative_to(project_root)
    except ValueError:
        relative = path
    return relative.as_posix()
