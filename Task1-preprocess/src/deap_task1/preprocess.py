from __future__ import annotations

import mne
import numpy as np
from mne.preprocessing import ICA, annotate_muscle_zscore

from deap_task1.constants import (
    DEAP_EOG_CHANNELS,
    DEFAULT_FILTER_HIGH,
    DEFAULT_FILTER_LOW,
)


def filter_and_reference_raw(
    raw: mne.io.BaseRaw,
    low_cut: float = DEFAULT_FILTER_LOW,
    high_cut: float = DEFAULT_FILTER_HIGH,
) -> mne.io.BaseRaw:
    filtered = raw.copy().load_data()
    picks = mne.pick_types(filtered.info, eeg=True, eog=True)
    filtered.filter(low_cut, high_cut, picks=picks, verbose="ERROR")
    filtered.set_eeg_reference("average", verbose="ERROR")
    return filtered


def fit_and_apply_ica(raw: mne.io.BaseRaw) -> tuple[mne.io.BaseRaw, ICA, dict[str, list[float]]]:
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(eeg_picks) == 0:
        raise ValueError("No EEG channels available for ICA")

    ica = ICA(
        n_components=min(20, len(eeg_picks)),
        method="fastica",
        random_state=97,
        max_iter="auto",
    )
    ica.fit(raw, picks=eeg_picks, decim=4, verbose="ERROR")

    excluded: set[int] = set()
    eog_scores: dict[str, list[float]] = {}
    for channel in DEAP_EOG_CHANNELS:
        component_ids, scores = ica.find_bads_eog(raw, ch_name=channel, verbose="ERROR")
        eog_scores[channel] = [float(score) for score in scores]
        excluded.update(component_ids)

    ica.exclude = sorted(excluded)
    cleaned = ica.apply(raw.copy(), verbose="ERROR")
    return cleaned, ica, eog_scores


def annotate_muscle_activity(
    raw: mne.io.BaseRaw,
    threshold: float = 4.0,
) -> tuple[mne.io.BaseRaw, dict[str, float | int | list[float]]]:
    annotations, scores = annotate_muscle_zscore(
        raw.copy(),
        ch_type="eeg",
        threshold=threshold,
        min_length_good=0.2,
        filter_freq=(30.0, 45.0),
        verbose="ERROR",
    )
    cleaned = raw.copy()
    cleaned.set_annotations(cleaned.annotations + annotations)
    log = {
        "muscle_annotation_count": len(annotations),
        "muscle_annotation_total_seconds": float(np.sum(annotations.duration)),
        "muscle_score_mean": float(np.mean(scores)) if len(scores) else None,
        "muscle_score_std": float(np.std(scores)) if len(scores) else None,
    }
    return cleaned, log
