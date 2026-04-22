from __future__ import annotations

import importlib
from pathlib import Path

import mne
from mne.preprocessing import ICA
import numpy as np
import pandas as pd
import pytest


N_TRIALS = 40
BASELINE_SECONDS = 3.0
VIDEO_SECONDS = 60.0
RAW_SFREQ = 128.0
TARGET_SFREQ = 64.0
VIDEO_CODE = 4
VIDEO_END_CODE = 5


def load_public_attr(module_name: str, attr_name: str):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"缺少模块 {module_name}: {exc}")

    try:
        return getattr(module, attr_name)
    except AttributeError:
        pytest.fail(f"模块 {module_name} 缺少公开符号 {attr_name}")


def pick_column(frame: pd.DataFrame, *candidates: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    pytest.fail(f"结果表缺少列，至少应包含 {candidates} 之一；当前列: {list(frame.columns)}")


def pick_layout_path(layout: dict[str, Path], token: str) -> Path:
    if token in layout:
        return Path(layout[token])

    token = token.lower()
    for key, value in layout.items():
        key_text = str(key).lower()
        path = Path(value)
        parts = {part.lower() for part in path.parts}
        if token in key_text or token in parts:
            return path

    pytest.fail(f"输出布局里找不到与 {token!r} 对应的路径，当前键: {list(layout)}")


@pytest.fixture(scope="module")
def synthetic_protocol():
    trial_spacing_seconds = VIDEO_SECONDS + 5.0
    duration_seconds = (
        BASELINE_SECONDS
        + (N_TRIALS - 1) * trial_spacing_seconds
        + VIDEO_SECONDS
        + 2.0
    )
    n_samples = int(duration_seconds * RAW_SFREQ)
    times = np.arange(n_samples, dtype=float) / RAW_SFREQ

    data = np.vstack(
        [
            np.sin(2.0 * np.pi * 10.0 * times),
            np.cos(2.0 * np.pi * 12.0 * times),
            np.sin(2.0 * np.pi * 8.0 * times + 0.5),
            np.cos(2.0 * np.pi * 6.0 * times + 1.0),
        ]
    )
    info = mne.create_info(
        ch_names=["Fz", "Cz", "Pz", "Oz"],
        sfreq=RAW_SFREQ,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    start_samples = np.round(
        (BASELINE_SECONDS + np.arange(N_TRIALS) * trial_spacing_seconds) * RAW_SFREQ
    ).astype(int)
    end_samples = start_samples + int(VIDEO_SECONDS * RAW_SFREQ)

    events = np.empty((N_TRIALS * 2, 3), dtype=int)
    events[0::2, 0] = start_samples
    events[0::2, 1] = 0
    events[0::2, 2] = VIDEO_CODE
    events[1::2, 0] = end_samples
    events[1::2, 1] = 0
    events[1::2, 2] = VIDEO_END_CODE

    video_events = events[0::2].copy()
    return raw, events, video_events


@pytest.fixture(scope="module")
def fitted_ica_for_plotting():
    sfreq = 128.0
    times = np.arange(int(10 * sfreq), dtype=float) / sfreq
    data = np.vstack(
        [
            np.sin(2.0 * np.pi * 10.0 * times),
            np.cos(2.0 * np.pi * 12.0 * times),
            np.sin(2.0 * np.pi * 8.0 * times + 0.5),
            np.cos(2.0 * np.pi * 6.0 * times + 1.0),
        ]
    )
    info = mne.create_info(
        ch_names=["Fz", "Cz", "Pz", "Oz"],
        sfreq=sfreq,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_montage("standard_1020", verbose="ERROR")

    ica = ICA(n_components=4, method="fastica", random_state=97, max_iter="auto")
    ica.fit(raw, verbose="ERROR")
    eog_scores = {
        "EXG1": [0.1, 0.8, 0.2, 0.4],
        "EXG2": [0.2, 0.7, 0.3, 0.5],
        "EXG3": [0.05, 0.1, 0.6, 0.2],
        "EXG4": [0.1, 0.2, 0.4, 0.3],
    }
    return ica, eog_scores


def test_extract_subject_id_from_bdf_name():
    extract_subject_id = load_public_attr("deap_task1.io", "extract_subject_id")

    assert extract_subject_id("s01.bdf") == "s01"
    assert extract_subject_id(r"nested\subject\s01.bdf") == "s01"


def test_extract_video_trial_events_parses_40_trials(synthetic_protocol):
    _, events, _ = synthetic_protocol
    extract_video_trial_events = load_public_attr(
        "deap_task1.segment",
        "extract_video_trial_events",
    )

    trials = extract_video_trial_events(
        events=events,
        sfreq=RAW_SFREQ,
        baseline_seconds=BASELINE_SECONDS,
        video_code=VIDEO_CODE,
        video_end_code=VIDEO_END_CODE,
    )

    assert isinstance(trials, pd.DataFrame)
    assert len(trials) == N_TRIALS

    start_col = pick_column(
        trials,
        "start_sample",
        "video_start_sample",
        "onset_sample",
        "event_sample",
    )
    end_col = pick_column(
        trials,
        "end_sample",
        "video_end_sample",
        "offset_sample",
    )
    baseline_start_col = pick_column(
        trials,
        "baseline_start_sample",
        "baseline_onset_sample",
        "prestim_start_sample",
    )
    duration_col = pick_column(
        trials,
        "video_duration_seconds",
        "duration_seconds",
        "video_seconds",
    )
    baseline_col = pick_column(
        trials,
        "baseline_seconds",
        "baseline_duration_seconds",
        "prestim_seconds",
    )

    expected_start = events[0::2, 0]
    expected_end = events[1::2, 0]
    expected_baseline_start = expected_start - int(BASELINE_SECONDS * RAW_SFREQ)

    assert np.array_equal(trials[start_col].to_numpy(), expected_start)
    assert np.array_equal(trials[end_col].to_numpy(), expected_end)
    assert np.array_equal(trials[baseline_start_col].to_numpy(), expected_baseline_start)
    assert np.allclose(trials[duration_col].to_numpy(), VIDEO_SECONDS)
    assert np.allclose(trials[baseline_col].to_numpy(), BASELINE_SECONDS)


def test_create_trial_epochs_outputs_60_second_epochs(synthetic_protocol):
    raw, _, video_events = synthetic_protocol
    create_trial_epochs = load_public_attr("deap_task1.segment", "create_trial_epochs")

    epochs = create_trial_epochs(
        raw=raw.copy(),
        video_events=video_events,
        target_sfreq=TARGET_SFREQ,
        baseline_seconds=BASELINE_SECONDS,
        video_seconds=VIDEO_SECONDS,
        picks="eeg",
    )

    expected_samples = int(TARGET_SFREQ * VIDEO_SECONDS)

    assert isinstance(epochs, mne.Epochs)
    assert len(epochs) == N_TRIALS
    assert epochs.info["sfreq"] == pytest.approx(TARGET_SFREQ)
    assert set(epochs.get_channel_types()) == {"eeg"}
    assert epochs.get_data(copy=False).shape == (N_TRIALS, 4, expected_samples)
    assert epochs.times[0] == pytest.approx(0.0)
    assert epochs.times[-1] == pytest.approx((expected_samples - 1) / TARGET_SFREQ)


def test_build_output_layout_creates_expected_directories(tmp_path: Path):
    build_output_layout = load_public_attr("deap_task1.io", "build_output_layout")

    layout = build_output_layout(project_root=tmp_path, subject_id="s01")

    assert isinstance(layout, dict)

    for token in ("interim", "processed", "figures"):
        path = pick_layout_path(layout, token)
        assert isinstance(path, Path)
        assert path.is_dir()
        assert path.exists()
        assert tmp_path == path or tmp_path in path.parents


def test_save_json_accepts_numpy_scalars(tmp_path: Path):
    save_json = load_public_attr("deap_task1.io", "save_json")

    target = tmp_path / "artifact_log.json"
    payload = {
        "count": np.int64(3),
        "seconds": np.float64(1.5),
        "items": [np.int64(1), np.float64(2.0)],
    }

    save_json(payload, target)

    saved = target.read_text(encoding="utf-8")
    assert '"count": 3' in saved
    assert '"seconds": 1.5' in saved


def test_extract_video_trial_events_rejects_non_alternating_events():
    extract_video_trial_events = load_public_attr(
        "deap_task1.segment",
        "extract_video_trial_events",
    )

    broken_events = np.array(
        [
            [384, 0, VIDEO_CODE],
            [640, 0, VIDEO_CODE],
            [8064, 0, VIDEO_END_CODE],
            [16000, 0, VIDEO_END_CODE],
        ],
        dtype=int,
    )

    with pytest.raises(ValueError):
        extract_video_trial_events(
            events=broken_events,
            sfreq=RAW_SFREQ,
            baseline_seconds=BASELINE_SECONDS,
            video_code=VIDEO_CODE,
            video_end_code=VIDEO_END_CODE,
        )


def test_extract_video_trial_events_ignores_leading_unmatched_end_events():
    extract_video_trial_events = load_public_attr(
        "deap_task1.segment",
        "extract_video_trial_events",
    )

    events = np.array(
        [
            [100, 0, VIDEO_END_CODE],
            [120, 0, VIDEO_END_CODE],
            [200, 0, VIDEO_CODE],
            [500, 0, VIDEO_END_CODE],
        ],
        dtype=int,
    )

    trials = extract_video_trial_events(
        events=events,
        sfreq=100.0,
        baseline_seconds=1.0,
        video_code=VIDEO_CODE,
        video_end_code=VIDEO_END_CODE,
    )

    assert len(trials) == 1
    assert int(trials.loc[0, "video_start_sample"]) == 200
    assert int(trials.loc[0, "video_end_sample"]) == 500


def test_create_trial_epochs_uses_fixed_window_even_if_end_markers_drift(synthetic_protocol):
    raw, events, video_events = synthetic_protocol
    extract_video_trial_events = load_public_attr(
        "deap_task1.segment",
        "extract_video_trial_events",
    )
    create_trial_epochs = load_public_attr("deap_task1.segment", "create_trial_epochs")

    trial_table = extract_video_trial_events(
        events=events,
        sfreq=RAW_SFREQ,
        baseline_seconds=BASELINE_SECONDS,
        video_code=VIDEO_CODE,
        video_end_code=VIDEO_END_CODE,
    )
    trial_table.loc[0, "video_duration_seconds"] = VIDEO_SECONDS + 2.0

    epochs = create_trial_epochs(
        raw=raw.copy(),
        video_events=video_events,
        target_sfreq=TARGET_SFREQ,
        baseline_seconds=BASELINE_SECONDS,
        video_seconds=VIDEO_SECONDS,
        picks="eeg",
        trial_table=trial_table,
    )

    assert len(epochs) == N_TRIALS


def test_create_trial_epochs_keeps_trials_with_muscle_annotations(synthetic_protocol):
    raw, events, video_events = synthetic_protocol
    extract_video_trial_events = load_public_attr(
        "deap_task1.segment",
        "extract_video_trial_events",
    )
    create_trial_epochs = load_public_attr("deap_task1.segment", "create_trial_epochs")

    trial_table = extract_video_trial_events(
        events=events,
        sfreq=RAW_SFREQ,
        baseline_seconds=BASELINE_SECONDS,
        video_code=VIDEO_CODE,
        video_end_code=VIDEO_END_CODE,
    )
    first_trial_onset = trial_table.loc[0, "video_start_sample"] / RAW_SFREQ
    raw.set_annotations(
        mne.Annotations(onset=[first_trial_onset + 1.0], duration=[2.0], description=["BAD_muscle"])
    )

    epochs = create_trial_epochs(
        raw=raw.copy(),
        video_events=video_events,
        target_sfreq=TARGET_SFREQ,
        baseline_seconds=BASELINE_SECONDS,
        video_seconds=VIDEO_SECONDS,
        picks="eeg",
        trial_table=trial_table,
    )

    assert len(epochs) == N_TRIALS


def test_positive_int_rejects_non_positive_values():
    positive_int = load_public_attr("run_task1", "positive_int")

    assert positive_int("2") == 2
    with pytest.raises(Exception):
        positive_int("0")
    with pytest.raises(Exception):
        positive_int("-1")


def test_prepare_waveform_segment_centers_and_scales_microvolts(synthetic_protocol):
    raw, _, _ = synthetic_protocol
    prepare_waveform_segment = load_public_attr(
        "deap_task1.plotting",
        "prepare_waveform_segment",
    )

    times, values = prepare_waveform_segment(
        raw=raw,
        channel="Cz",
        start_seconds=1.0,
        duration_seconds=0.5,
    )

    assert times[0] == pytest.approx(1.0)
    assert times[-1] <= 1.5
    assert np.mean(values) == pytest.approx(0.0, abs=1e-6)
    assert np.max(np.abs(values)) > 1e5


def test_summarize_eog_components_ranks_unique_components(fitted_ica_for_plotting):
    _, eog_scores = fitted_ica_for_plotting
    summarize_eog_components = load_public_attr(
        "deap_task1.plotting",
        "summarize_eog_components",
    )

    summary = summarize_eog_components(eog_scores, top_n=3)

    assert [item["component_index"] for item in summary] == [1, 2, 3]
    assert summary[0]["channel"] == "EXG1"
    assert summary[0]["score"] == pytest.approx(0.8)


def test_save_ica_topomap_writes_png(tmp_path: Path, fitted_ica_for_plotting):
    ica, eog_scores = fitted_ica_for_plotting
    save_ica_topomap = load_public_attr("deap_task1.plotting", "save_ica_topomap")

    target = tmp_path / "ica_eog_topomap.png"
    save_ica_topomap(ica=ica, eog_scores=eog_scores, output_path=target, top_n=3)

    assert target.exists()
    assert target.stat().st_size > 0


def test_get_topomap_colorbar_rect_places_bar_on_right_margin():
    get_topomap_colorbar_rect = load_public_attr(
        "deap_task1.plotting",
        "get_topomap_colorbar_rect",
    )

    left, bottom, width, height = get_topomap_colorbar_rect(3)

    assert left >= 0.92
    assert bottom == pytest.approx(0.18)
    assert width <= 0.03
    assert height == pytest.approx(0.6)


def test_resolve_status_channels_prefers_last_fallback_candidate():
    resolve_status_channels = load_public_attr("deap_task1.io", "resolve_status_channels")

    status_channel, auxiliary_channels = resolve_status_channels(
        [
            *["Fz", "Cz", "Pz", "Oz"],
            *["EXG1", "EXG2", "EXG3", "EXG4"],
            *["EXG5", "EXG6", "EXG7", "EXG8"],
            *["GSR1", "GSR2", "Erg1", "Erg2", "Resp", "Plet", "Temp"],
            "",
            "-0",
            "-1",
        ]
    )

    assert status_channel == "-1"
    assert auxiliary_channels == ["", "-0"]


def test_find_status_events_masks_high_bits():
    find_status_events = load_public_attr("deap_task1.io", "find_status_events")

    data = np.array(
        [[65280.0, 65280.0, 65284.0, 65284.0, 65280.0, 65285.0, 65285.0, 65280.0]]
    )
    info = mne.create_info(["Status"], sfreq=128.0, ch_types=["stim"])
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    events = find_status_events(raw)

    assert events[:, 2].tolist() == [4, 5]
