"""
features.py
-----------
R-peak detection and Heart Rate Variability (HRV) feature extraction.

HRV — variation in the time between consecutive heartbeats — is a
clinically significant marker. Reduced HRV is associated with cardiac
disease, stress, and autonomic nervous system dysfunction.
"""

import numpy as np
from scipy.signal import find_peaks
from typing import Tuple, Dict


def detect_r_peaks(
    filtered_ecg: np.ndarray,
    fs: float,
    min_rr_s: float = 0.6,
    prominence_factor: float = 0.3,
) -> np.ndarray:
    """
    Detect R-peaks (QRS complex peaks) in a pre-filtered ECG signal.

    Uses scipy's find_peaks with physiologically motivated constraints:
    - Height threshold based on signal mean + scaled std (adaptive).
    - Minimum distance between peaks (refractory period: ~600ms at max HR 100 bpm).

    Args:
        filtered_ecg:      Bandpass-filtered ECG signal.
        fs:                Sampling frequency in Hz.
        min_rr_s:          Minimum RR interval in seconds (default 0.6 = 100 bpm max HR).
        prominence_factor: Scales std to set adaptive height threshold.

    Returns:
        Array of sample indices where R-peaks occur.
    """
    height_threshold = np.mean(filtered_ecg) + prominence_factor * np.std(filtered_ecg)
    min_distance = int(min_rr_s * fs)

    peaks, _ = find_peaks(
        filtered_ecg,
        height=height_threshold,
        distance=min_distance,
    )
    return peaks


def compute_rr_intervals(peaks: np.ndarray, fs: float) -> np.ndarray:
    """
    Convert R-peak sample indices to RR intervals in milliseconds.

    Args:
        peaks: Sample indices of R-peaks.
        fs:    Sampling frequency in Hz.

    Returns:
        Array of RR intervals in milliseconds.
    """
    return np.diff(peaks) / fs * 1000.0


def extract_hrv_features(rr_ms: np.ndarray) -> Dict[str, float]:
    """
    Extract time-domain HRV features from RR interval series.

    Features:
        mean_rr:    Mean RR interval (ms). Inversely related to heart rate.
        mean_hr:    Mean heart rate (bpm).
        sdnn:       Std dev of RR intervals. Overall HRV measure.
        rmssd:      Root mean square of successive differences.
                    Reflects parasympathetic (vagal) modulation.
        pnn50:      % of successive RR differences > 50 ms.
                    Another vagal tone marker. Low pNN50 → stress / pathology.
        cv_rr:      Coefficient of variation (SDNN / mean RR × 100).
                    Normalized HRV, useful for comparing across heart rates.

    Args:
        rr_ms: RR interval series in milliseconds.

    Returns:
        Dictionary of HRV feature names to float values.
    """
    if len(rr_ms) < 2:
        raise ValueError("Need at least 2 RR intervals to compute HRV features.")

    successive_diffs = np.diff(rr_ms)

    features = {
        "mean_rr_ms":  float(np.mean(rr_ms)),
        "mean_hr_bpm": float(60_000.0 / np.mean(rr_ms)),
        "sdnn_ms":     float(np.std(rr_ms, ddof=1)),
        "rmssd_ms":    float(np.sqrt(np.mean(successive_diffs ** 2))),
        "pnn50_pct":   float(np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100),
        "cv_rr_pct":   float(np.std(rr_ms, ddof=1) / np.mean(rr_ms) * 100),
    }
    return features


def segment_beats(
    ecg: np.ndarray,
    peaks: np.ndarray,
    fs: float,
    pre_ms: float = 200.0,
    post_ms: float = 400.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice fixed-length windows around each R-peak to produce beat segments.

    Each segment is centred on the R-peak with `pre_ms` ms before and
    `post_ms` ms after. Beats that would require samples outside the
    signal boundary are discarded.

    This is the input preparation step for beat-level ML classifiers.

    Args:
        ecg:     Filtered ECG signal.
        peaks:   R-peak sample indices.
        fs:      Sampling frequency in Hz.
        pre_ms:  Milliseconds before the R-peak to include.
        post_ms: Milliseconds after the R-peak to include.

    Returns:
        segments: (N, L) array of beat waveforms where L = pre + post samples.
        valid_peaks: Subset of peak indices whose windows fit inside the signal.
    """
    pre_samples  = int(pre_ms  / 1000.0 * fs)
    post_samples = int(post_ms / 1000.0 * fs)
    n_samples    = pre_samples + post_samples

    segments = []
    valid_peaks = []

    for p in peaks:
        start = p - pre_samples
        end   = p + post_samples
        if start < 0 or end > len(ecg):
            continue
        segment = ecg[start:end]
        # Amplitude normalise each beat so shape, not scale, is the feature
        segment = (segment - segment.mean()) / (segment.std() + 1e-8)
        segments.append(segment)
        valid_peaks.append(p)

    return np.array(segments), np.array(valid_peaks)
