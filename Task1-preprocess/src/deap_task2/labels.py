from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from deap_task2.constants import (
    BINARY_THRESHOLD,
    PARTICIPANT_RATINGS_FILENAME,
    VIDEO_LIST_FILENAME,
)
from deap_task2.io import read_csv_from_source


def normalize_subject_id(value: object) -> str:
    text = str(value).strip()
    match = re.search(r"(\d+)", text)
    if match is None:
        raise ValueError(f"Could not parse subject id from value: {value!r}")
    return f"s{int(match.group(1)):02d}"


def score_to_binary(value: float, threshold: float = BINARY_THRESHOLD) -> int:
    return int(float(value) > threshold)


def encode_quadrant(valence_binary: int, arousal_binary: int) -> int:
    mapping = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3,
    }
    return mapping[(valence_binary, arousal_binary)]


def label_quadrant(valence_binary: int, arousal_binary: int) -> str:
    mapping = {
        (0, 0): "LVLA",
        (0, 1): "LVHA",
        (1, 0): "HVLA",
        (1, 1): "HVHA",
    }
    return mapping[(valence_binary, arousal_binary)]


def standardize_video_list(video_list: pd.DataFrame) -> pd.DataFrame:
    frame = video_list.copy()
    frame = frame[frame["Experiment_id"].notna()].copy()
    frame["Experiment_id"] = frame["Experiment_id"].astype(int)
    frame = frame.rename(
        columns={
            "Experiment_id": "experiment_id",
            "Lastfm_tag": "lastfm_tag",
            "Artist": "artist",
            "Title": "title",
            "Youtube_link": "youtube_link",
            "Highlight_start": "highlight_start",
        }
    )
    columns = [
        "experiment_id",
        "lastfm_tag",
        "artist",
        "title",
        "youtube_link",
        "highlight_start",
    ]
    return frame[columns].sort_values("experiment_id").reset_index(drop=True)


def standardize_participant_ratings(
    ratings: pd.DataFrame,
    video_list: pd.DataFrame | None = None,
    threshold: float = BINARY_THRESHOLD,
) -> pd.DataFrame:
    frame = ratings.rename(
        columns={
            "Participant_id": "participant_id",
            "Trial": "trial_index",
            "Experiment_id": "experiment_id",
            "Start_time": "start_time",
            "Valence": "valence",
            "Arousal": "arousal",
            "Dominance": "dominance",
            "Liking": "liking",
            "Familiarity": "familiarity",
        }
    ).copy()

    required = [
        "participant_id",
        "trial_index",
        "experiment_id",
        "start_time",
        "valence",
        "arousal",
        "dominance",
        "liking",
        "familiarity",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing rating columns: {missing}")

    frame["subject_id"] = frame["participant_id"].map(normalize_subject_id)
    frame["subject_number"] = frame["subject_id"].str.extract(r"(\d+)").astype(int)
    for column in ["trial_index", "experiment_id", "start_time"]:
        frame[column] = frame[column].astype(int)
    for column in ["valence", "arousal", "dominance", "liking", "familiarity"]:
        frame[column] = frame[column].astype(float)

    frame["valence_binary"] = frame["valence"].map(lambda value: score_to_binary(value, threshold))
    frame["arousal_binary"] = frame["arousal"].map(lambda value: score_to_binary(value, threshold))
    frame["dominance_binary"] = frame["dominance"].map(lambda value: score_to_binary(value, threshold))
    frame["quadrant_id"] = [
        encode_quadrant(valence_binary, arousal_binary)
        for valence_binary, arousal_binary in zip(
            frame["valence_binary"],
            frame["arousal_binary"],
            strict=True,
        )
    ]
    frame["quadrant_label"] = [
        label_quadrant(valence_binary, arousal_binary)
        for valence_binary, arousal_binary in zip(
            frame["valence_binary"],
            frame["arousal_binary"],
            strict=True,
        )
    ]

    if video_list is not None:
        frame = frame.merge(
            standardize_video_list(video_list),
            on="experiment_id",
            how="left",
            validate="m:1",
        )

    frame = frame.sort_values(["subject_number", "trial_index"]).reset_index(drop=True)
    ordered_columns = [
        "subject_id",
        "subject_number",
        "trial_index",
        "experiment_id",
        "start_time",
        "valence",
        "arousal",
        "dominance",
        "liking",
        "familiarity",
        "valence_binary",
        "arousal_binary",
        "dominance_binary",
        "quadrant_id",
        "quadrant_label",
    ]
    optional_columns = [
        "lastfm_tag",
        "artist",
        "title",
        "youtube_link",
        "highlight_start",
    ]
    ordered_columns.extend([column for column in optional_columns if column in frame.columns])
    return frame[ordered_columns]


def export_task2_labels(
    metadata_source: Path,
    output_path: Path,
    threshold: float = BINARY_THRESHOLD,
) -> pd.DataFrame:
    ratings = read_csv_from_source(metadata_source, PARTICIPANT_RATINGS_FILENAME)
    video_list = read_csv_from_source(metadata_source, VIDEO_LIST_FILENAME)
    labels = standardize_participant_ratings(ratings, video_list=video_list, threshold=threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output_path, index=False, encoding="utf-8")
    return labels
