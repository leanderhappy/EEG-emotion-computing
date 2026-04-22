from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import mne
import pandas as pd

from deap_task1.constants import (
    DEFAULT_BASELINE_SECONDS,
    DEFAULT_TARGET_SFREQ,
    DEFAULT_VIDEO_END_CODE,
    DEFAULT_VIDEO_SECONDS,
    DEFAULT_VIDEO_START_CODE,
)
from deap_task1.io import (
    build_output_layout,
    extract_subject_id,
    find_status_events,
    load_deap_raw,
    save_channel_table,
    save_dataframe,
    save_json,
)
from deap_task1.plotting import (
    save_eog_score_plot,
    save_ica_topomap,
    save_psd_comparison,
    save_waveform_comparison,
    summarize_eog_components,
)
from deap_task1.preprocess import annotate_muscle_activity, filter_and_reference_raw, fit_and_apply_ica
from deap_task1.segment import (
    create_trial_epochs,
    extract_video_trial_events,
    summarize_annotation_overlap,
)


@dataclass(frozen=True)
class PipelineResult:
    subject_id: str
    bdf_path: str
    epochs_path: str
    trial_table_path: str
    qc_table_path: str


def run_subject_pipeline(
    bdf_path: Path,
    project_root: Path,
    target_sfreq: float = DEFAULT_TARGET_SFREQ,
    baseline_seconds: float = DEFAULT_BASELINE_SECONDS,
    video_seconds: float = DEFAULT_VIDEO_SECONDS,
) -> PipelineResult:
    subject_id = extract_subject_id(str(bdf_path))
    layout = build_output_layout(project_root=project_root, subject_id=subject_id)

    raw = load_deap_raw(bdf_path, preload=True)
    events = find_status_events(raw)
    trial_table = extract_video_trial_events(
        events=events,
        sfreq=raw.info["sfreq"],
        baseline_seconds=baseline_seconds,
        video_code=DEFAULT_VIDEO_START_CODE,
        video_end_code=DEFAULT_VIDEO_END_CODE,
    )
    if len(trial_table) != 40:
        raise ValueError(f"Expected 40 DEAP trials, found {len(trial_table)}")
    save_dataframe(trial_table, layout["interim"] / "trial_events.csv")
    save_channel_table(raw, layout["interim"] / "channel_table.csv")

    video_events = events[events[:, 2] == DEFAULT_VIDEO_START_CODE]
    raw_before = raw.copy().pick("eeg")
    filtered = filter_and_reference_raw(raw)
    cleaned, ica, eog_scores = fit_and_apply_ica(filtered)
    cleaned, muscle_log = annotate_muscle_activity(cleaned)

    save_psd_comparison(raw_before, cleaned.copy().pick("eeg"), layout["figures"] / "psd_comparison.png")
    save_waveform_comparison(raw_before, cleaned.copy().pick("eeg"), layout["figures"] / "waveform_comparison.png")
    save_eog_score_plot(eog_scores, ica.exclude, layout["figures"] / "ica_eog_scores.png")
    save_ica_topomap(ica, eog_scores, layout["figures"] / "ica_eog_topomap.png")

    qc_table = summarize_annotation_overlap(
        trial_table=trial_table,
        annotations=cleaned.annotations,
        sfreq=raw.info["sfreq"],
        video_seconds=video_seconds,
        output_path=layout["processed"] / "trial_qc.csv",
    )

    epochs = create_trial_epochs(
        raw=cleaned,
        video_events=video_events,
        target_sfreq=target_sfreq,
        baseline_seconds=baseline_seconds,
        video_seconds=video_seconds,
        picks="eeg",
        trial_table=trial_table,
    )
    epochs.metadata = pd.concat([trial_table, qc_table[["muscle_overlap_seconds"]]], axis=1)
    epochs_path = layout["processed"] / "epochs_clean.fif"
    epochs.save(epochs_path, overwrite=True, verbose="ERROR")

    save_json(
        {
            "subject_id": subject_id,
            "bdf_path": str(bdf_path),
            "sample_rate_hz": raw.info["sfreq"],
            "target_sample_rate_hz": target_sfreq,
            "trial_count": len(trial_table),
            "ica_excluded_components": ica.exclude,
            "top_eog_components": summarize_eog_components(eog_scores, top_n=3),
            "muscle_log": muscle_log,
        },
        layout["interim"] / "artifact_log.json",
    )

    return PipelineResult(
        subject_id=subject_id,
        bdf_path=str(bdf_path),
        epochs_path=str(epochs_path),
        trial_table_path=str(layout["interim"] / "trial_events.csv"),
        qc_table_path=str(layout["processed"] / "trial_qc.csv"),
    )


def result_to_frame(results: list[PipelineResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(result) for result in results])
