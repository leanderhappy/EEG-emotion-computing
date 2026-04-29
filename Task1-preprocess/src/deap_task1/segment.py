from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from deap_task1.constants import DEFAULT_VIDEO_START_CODE


def extract_video_trial_events(
    events: np.ndarray,
    sfreq: float,
    baseline_seconds: float = 3.0,
    video_code: int = 4,
    video_end_code: int = 5,
) -> pd.DataFrame:
    events = np.asarray(events, dtype=int)
    if events.ndim != 2 or events.shape[1] != 3:
        raise ValueError("events must have shape (n_events, 3)")

    if not np.any(events[:, 2] == video_code):
        raise ValueError(f"No video start events found for code {video_code}")

    baseline_samples = int(round(baseline_seconds * sfreq))
    rows: list[dict[str, int | float]] = []
    start_samples = events[events[:, 2] == video_code, 0].astype(int)
    end_samples = events[events[:, 2] == video_end_code, 0].astype(int)
    end_index = 0

    for trial_index, start_sample in enumerate(start_samples, start=1):
        while end_index < len(end_samples) and end_samples[end_index] <= start_sample:
            end_index += 1
        if end_index >= len(end_samples):
            raise ValueError(f"Missing video end event for trial {trial_index}")

        end_sample = int(end_samples[end_index])
        next_start_sample = (
            int(start_samples[trial_index]) if trial_index < len(start_samples) else None
        )
        if next_start_sample is not None and end_sample >= next_start_sample:
            raise ValueError("Video start events must be strictly paired with end events")
        if end_sample <= start_sample:
            raise ValueError(
                f"Trial {trial_index} end sample {end_sample} is not after start {start_sample}"
            )
        baseline_start_sample = start_sample - baseline_samples
        if baseline_start_sample < 0:
            raise ValueError(
                f"Trial {trial_index} baseline would start before recording begins"
            )

        rows.append(
            {
                "trial_index": trial_index,
                "video_start_sample": start_sample,
                "video_end_sample": end_sample,
                "baseline_start_sample": baseline_start_sample,
                "baseline_seconds": float(baseline_seconds),
                "video_duration_seconds": float((end_sample - start_sample) / sfreq),
            }
        )
        end_index += 1

    if not rows:
        raise ValueError("No complete video trials could be extracted")

    return pd.DataFrame(rows)


def create_trial_epochs(
    raw: mne.io.BaseRaw,
    video_events: np.ndarray,
    target_sfreq: float,
    baseline_seconds: float = 3.0,
    video_seconds: float = 60.0,
    picks: str = "eeg",
    trial_table: pd.DataFrame | None = None,
) -> mne.Epochs:
    if target_sfreq <= 0:
        raise ValueError("target_sfreq must be positive")

    working_raw = raw.copy().load_data()
    video_events = np.asarray(video_events, dtype=int)
    if trial_table is not None:
        if len(trial_table) != len(video_events):
            raise ValueError("trial_table and video_events must describe the same number of trials")
        if not np.array_equal(trial_table["video_start_sample"].to_numpy(dtype=int), video_events[:, 0]):
            raise ValueError("trial_table video_start_sample does not match video_events")
        expected_window_samples = int(round(float(video_seconds) * raw.info["sfreq"]))
        expected_end_samples = (
            trial_table["video_start_sample"].to_numpy(dtype=int) + expected_window_samples
        )
        if np.any(expected_end_samples > raw.n_times):
            raise ValueError("Fixed 60-second video window extends beyond the recording")
        if len(trial_table) > 1:
            start_spacing = np.diff(trial_table["video_start_sample"].to_numpy(dtype=int))
            if np.any(start_spacing <= expected_window_samples):
                raise ValueError("Video start events are too close together for fixed 60-second windows")

    working_raw, resampled_events = working_raw.resample(
        target_sfreq,
        events=video_events,
        verbose="ERROR",
    )

    tmax = video_seconds - (1.0 / target_sfreq)
    event_id = {"video": int(video_events[0, 2]) if len(video_events) else DEFAULT_VIDEO_START_CODE}
    epochs = mne.Epochs(
        working_raw,
        resampled_events,
        event_id=event_id,
        tmin=-baseline_seconds,
        tmax=tmax,
        baseline=(-baseline_seconds, 0.0),
        picks=picks,
        preload=True,
        reject_by_annotation=False,
        event_repeated="drop",
        verbose="ERROR",
    )
    epochs.crop(tmin=0.0, tmax=tmax)
    return epochs


def summarize_annotation_overlap(
    trial_table: pd.DataFrame,
    annotations: mne.Annotations,
    sfreq: float,
    video_seconds: float | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for row in trial_table.itertuples(index=False):
        trial_start_seconds = row.video_start_sample / sfreq
        if video_seconds is None:
            trial_end_seconds = row.video_end_sample / sfreq
        else:
            trial_end_seconds = trial_start_seconds + float(video_seconds)
        overlap_seconds = 0.0
        for onset, duration, description in zip(
            annotations.onset,
            annotations.duration,
            annotations.description,
        ):
            if "muscle" not in description.lower():
                continue
            left = max(trial_start_seconds, onset)
            right = min(trial_end_seconds, onset + duration)
            overlap_seconds += max(0.0, right - left)
        rows.append(
            {
                "trial_index": int(row.trial_index),
                "muscle_overlap_seconds": overlap_seconds,
            }
        )

    frame = pd.DataFrame(rows)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False, encoding="utf-8")
    return frame
