from __future__ import annotations

DEAP_EEG_CHANNELS = [
    "Fp1",
    "AF3",
    "F7",
    "F3",
    "FC1",
    "FC5",
    "T7",
    "C3",
    "CP1",
    "CP5",
    "P7",
    "P3",
    "Pz",
    "PO3",
    "O1",
    "Oz",
    "O2",
    "PO4",
    "P4",
    "P8",
    "CP6",
    "CP2",
    "C4",
    "T8",
    "FC6",
    "FC2",
    "F4",
    "F8",
    "AF4",
    "Fp2",
    "Fz",
    "Cz",
]

DEAP_EOG_CHANNELS = ["EXG1", "EXG2", "EXG3", "EXG4"]
DEAP_EMG_CHANNELS = ["EXG5", "EXG6", "EXG7", "EXG8"]
DEAP_MISC_CHANNELS = ["GSR1", "GSR2", "Erg1", "Erg2", "Resp", "Plet", "Temp"]
DEAP_STATUS_CHANNEL = "Status"

DEFAULT_TARGET_SFREQ = 128.0
DEFAULT_BASELINE_SECONDS = 3.0
DEFAULT_VIDEO_SECONDS = 60.0
DEFAULT_FILTER_LOW = 4.0
DEFAULT_FILTER_HIGH = 45.0
DEFAULT_VIDEO_START_CODE = 4
DEFAULT_VIDEO_END_CODE = 5
