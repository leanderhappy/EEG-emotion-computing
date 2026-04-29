from __future__ import annotations

import json
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from deap_task1.constants import (
    DEAP_EEG_CHANNELS,
    DEAP_EMG_CHANNELS,
    DEAP_EOG_CHANNELS,
    DEAP_MISC_CHANNELS,
    DEAP_STATUS_CHANNEL,
)

SUBJECT_PATTERN = re.compile(r"(s\d{2})\.bdf$", re.IGNORECASE)
KNOWN_DEAP_CHANNELS = set(
    DEAP_EEG_CHANNELS
    + DEAP_EOG_CHANNELS
    + DEAP_EMG_CHANNELS
    + DEAP_MISC_CHANNELS
    + [DEAP_STATUS_CHANNEL]
)


def extract_subject_id(path: str) -> str:
    match = SUBJECT_PATTERN.search(Path(path).name)
    if match is None:
        raise ValueError(f"Cannot extract DEAP subject id from: {path}")
    return match.group(1).lower()


def resolve_default_data_root(project_root: Path) -> Path:
    return project_root.parent / "data" / "raw" / "DEAP" / "data_original"


def build_output_layout(project_root: Path, subject_id: str) -> dict[str, Path]:
    project_root = Path(project_root).resolve()
    interim_dir = project_root / "data" / "interim" / subject_id
    processed_dir = project_root / "data" / "processed" / subject_id
    figures_dir = project_root / "output" / "figures" / subject_id

    for path in (interim_dir, processed_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    return {
        "project_root": project_root,
        "interim": interim_dir,
        "processed": processed_dir,
        "figures": figures_dir,
    }


def list_subject_files(data_root: Path, subject_id: str | None = None) -> list[Path]:
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"DEAP data root not found: {data_root}")

    files = sorted(data_root.glob("s*.bdf"))
    if not files:
        raise FileNotFoundError(f"No BDF files found under: {data_root}")

    if subject_id is None:
        return files

    subject_id = subject_id.lower()
    selected = [path for path in files if extract_subject_id(path.name) == subject_id]
    if not selected:
        raise FileNotFoundError(f"Subject {subject_id} not found under: {data_root}")
    return selected


def resolve_status_channels(ch_names: list[str]) -> tuple[str, list[str]]:
    if DEAP_STATUS_CHANNEL in ch_names:
        return DEAP_STATUS_CHANNEL, []

    fallback_candidates = [name for name in ch_names if not str(name).strip() or str(name).startswith("-")]
    if fallback_candidates:
        return fallback_candidates[-1], fallback_candidates[:-1]

    unknown_candidates = [name for name in ch_names if name not in KNOWN_DEAP_CHANNELS]
    if unknown_candidates:
        return unknown_candidates[-1], unknown_candidates[:-1]

    raise ValueError("Missing expected DEAP status channel")


def find_status_events(raw: mne.io.BaseRaw, shortest_event: int = 1) -> np.ndarray:
    return mne.find_events(
        raw,
        stim_channel=DEAP_STATUS_CHANNEL,
        shortest_event=shortest_event,
        mask=255,
        mask_type="and",
        verbose="ERROR",
    )


def configure_deap_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    status_channel, auxiliary_status_channels = resolve_status_channels(raw.ch_names)
    rename_map = {}
    if status_channel != DEAP_STATUS_CHANNEL:
        rename_map[status_channel] = DEAP_STATUS_CHANNEL
    for index, channel in enumerate(auxiliary_status_channels, start=1):
        rename_map[channel] = f"Status_aux_{index}"
    if rename_map:
        raw.rename_channels(rename_map)

    channel_types = {
        **{name: "eeg" for name in DEAP_EEG_CHANNELS},
        **{name: "eog" for name in DEAP_EOG_CHANNELS},
        **{name: "emg" for name in DEAP_EMG_CHANNELS},
        **{name: "misc" for name in DEAP_MISC_CHANNELS},
        DEAP_STATUS_CHANNEL: "stim",
        **{name: "misc" for name in rename_map.values() if name != DEAP_STATUS_CHANNEL},
    }
    missing = sorted(name for name in channel_types if name not in raw.ch_names)
    if missing:
        raise ValueError(f"Missing expected DEAP channels: {missing}")

    raw.set_channel_types(channel_types, verbose="ERROR")
    raw.set_montage("biosemi32", on_missing="raise", verbose="ERROR")
    return raw


def load_deap_raw(bdf_path: Path, preload: bool = True) -> mne.io.BaseRaw:
    probe = mne.io.read_raw_bdf(
        str(bdf_path),
        preload=False,
        stim_channel=None,
        verbose="ERROR",
    )
    status_channel, _ = resolve_status_channels(probe.ch_names)
    probe.close()
    raw = mne.io.read_raw_bdf(
        str(bdf_path),
        preload=preload,
        stim_channel=status_channel,
        verbose="ERROR",
    )
    return configure_deap_channels(raw)


def save_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def save_channel_table(raw: mne.io.BaseRaw, path: Path) -> None:
    frame = pd.DataFrame({"channel": raw.ch_names, "type": raw.get_channel_types()})
    save_dataframe(frame, path)


def _to_jsonable(value):
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
