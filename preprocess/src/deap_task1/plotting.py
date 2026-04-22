from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.viz import plot_topomap


def prepare_waveform_segment(
    raw: mne.io.BaseRaw,
    channel: str,
    start_seconds: float,
    duration_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    start_sample = int(start_seconds * raw.info["sfreq"])
    stop_sample = int((start_seconds + duration_seconds) * raw.info["sfreq"])
    data, times = raw.copy().pick([channel]).get_data(
        start=start_sample,
        stop=stop_sample,
        return_times=True,
    )
    centered_microvolts = (data[0] - np.mean(data[0])) * 1e6
    return times, centered_microvolts


def summarize_eog_components(
    eog_scores: dict[str, list[float]],
    top_n: int = 3,
) -> list[dict[str, float | int | str]]:
    best_by_component: dict[int, dict[str, float | int | str]] = {}
    for channel, scores in eog_scores.items():
        score_array = np.asarray(scores, dtype=float)
        for component_index, score in enumerate(score_array):
            candidate = {
                "component_index": int(component_index),
                "channel": channel,
                "score": float(score),
                "abs_score": float(abs(score)),
            }
            current = best_by_component.get(component_index)
            if current is None or candidate["abs_score"] > current["abs_score"]:
                best_by_component[component_index] = candidate

    ranked = sorted(
        best_by_component.values(),
        key=lambda item: (-float(item["abs_score"]), int(item["component_index"])),
    )
    trimmed = ranked[: max(0, top_n)]
    return [
        {
            "component_index": int(item["component_index"]),
            "channel": str(item["channel"]),
            "score": float(item["score"]),
        }
        for item in trimmed
    ]


def get_topomap_colorbar_rect(n_components: int) -> tuple[float, float, float, float]:
    if n_components <= 0:
        raise ValueError("n_components must be positive")

    return (0.92, 0.18, 0.02, 0.6)


def save_psd_comparison(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    output_path: Path,
) -> None:
    spec_before = raw_before.compute_psd(
        fmin=1.0,
        fmax=60.0,
        picks="eeg",
        n_fft=256,
        n_per_seg=256,
        verbose="ERROR",
    )
    spec_after = raw_after.compute_psd(
        fmin=1.0,
        fmax=60.0,
        picks="eeg",
        n_fft=256,
        n_per_seg=256,
        verbose="ERROR",
    )
    psd_before, freqs = spec_before.get_data(return_freqs=True)
    psd_after, _ = spec_after.get_data(return_freqs=True)

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(freqs, 10.0 * np.log10(psd_before.mean(axis=0)), label="Before")
    axis.plot(freqs, 10.0 * np.log10(psd_after.mean(axis=0)), label="After")
    axis.set_title("PSD comparison")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Power (dB)")
    axis.legend()
    axis.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_waveform_comparison(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    output_path: Path,
    channels: tuple[str, ...] = ("Fp1", "Cz", "Oz"),
    start_seconds: float = 60.0,
    duration_seconds: float = 2.0,
) -> None:
    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 7), sharex=True)
    if len(channels) == 1:
        axes = [axes]
    for axis, channel in zip(axes, channels):
        times, before = prepare_waveform_segment(
            raw=raw_before,
            channel=channel,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )
        _, after = prepare_waveform_segment(
            raw=raw_after,
            channel=channel,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )
        amplitude_limit = max(np.max(np.abs(before)), np.max(np.abs(after))) * 1.1
        axis.plot(times, before, label="Before", linewidth=1.3, alpha=0.8)
        axis.plot(times, after, label="After", linewidth=1.3, alpha=0.8)
        axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        axis.set_ylim(-amplitude_limit, amplitude_limit)
        axis.set_ylabel(f"{channel}\n(uV)")
        axis.grid(True, alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("Seconds")
    fig.suptitle("Waveform comparison (centered, microvolts)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_eog_score_plot(
    eog_scores: dict[str, list[float]],
    excluded_components: list[int],
    output_path: Path,
    top_n: int = 3,
) -> None:
    fig, axis = plt.subplots(figsize=(10, 5))
    for channel, scores in eog_scores.items():
        if not scores:
            continue
        axis.plot(scores, label=channel)
    for item in summarize_eog_components(eog_scores, top_n=top_n):
        component_index = int(item["component_index"])
        score = float(item["score"])
        axis.scatter(component_index, score, color="black", s=35, zorder=4)
        axis.annotate(
            f"C{component_index}\n{item['channel']}",
            (component_index, score),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )
    axis.set_title("ICA EOG component scores")
    axis.set_xlabel("Component index")
    axis.set_ylabel("Score")
    for component in excluded_components:
        axis.axvline(component, color="red", linestyle="--", alpha=0.3)
    axis.legend()
    axis.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_ica_topomap(
    ica: mne.preprocessing.ICA,
    eog_scores: dict[str, list[float]],
    output_path: Path,
    top_n: int = 3,
) -> None:
    summary = summarize_eog_components(eog_scores, top_n=top_n)
    if not summary:
        raise ValueError("No EOG component scores available for ICA topomap plotting")

    component_maps = ica.get_components()
    eeg_info = mne.pick_info(ica.info, mne.pick_types(ica.info, eeg=True))
    colorbar_rect = get_topomap_colorbar_rect(len(summary))

    fig, axes = plt.subplots(1, len(summary), figsize=(4 * len(summary), 4), squeeze=False)
    axes_flat = axes.ravel()
    last_image = None
    for axis, item in zip(axes_flat, summary):
        component_index = int(item["component_index"])
        component_values = component_maps[:, component_index]
        last_image, _ = plot_topomap(
            component_values,
            eeg_info,
            axes=axis,
            show=False,
            contours=0,
            sensors=True,
        )
        axis.set_title(
            f"C{component_index}\n{item['channel']} score={float(item['score']):.2f}",
            fontsize=10,
        )

    if last_image is not None:
        cax = fig.add_axes(colorbar_rect)
        fig.colorbar(last_image, cax=cax)
    fig.suptitle("Top EOG-like ICA component maps")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.78, right=colorbar_rect[0] - 0.04, wspace=0.35)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
