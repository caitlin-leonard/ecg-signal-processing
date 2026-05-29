"""
filters.py
----------
A collection of digital filters for ECG signal preprocessing.

Each function is stateless and returns a filtered numpy array.
All filters use zero-phase (filtfilt) processing where possible to
avoid introducing time-delay artifacts that would shift QRS complexes.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, lfilter, freqz


def butter_lowpass(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Butterworth low-pass filter.

    Removes high-frequency noise (muscle artifacts, electrode motion).
    Typical cutoff for ECG: 40 Hz — everything above is noise.

    Args:
        data:   Raw ECG signal.
        cutoff: -3 dB cutoff frequency in Hz.
        fs:     Sampling frequency in Hz.
        order:  Filter order (higher = steeper rolloff, more ringing).

    Returns:
        Zero-phase filtered signal.
    """
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, data)


def butter_highpass(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Butterworth high-pass filter.

    Removes baseline wander caused by respiration, electrode movement,
    and slow DC drift. Typical cutoff: 0.5 Hz.

    Args:
        data:   ECG signal (ideally already low-pass filtered).
        cutoff: -3 dB cutoff frequency in Hz.
        fs:     Sampling frequency in Hz.
        order:  Filter order.

    Returns:
        Zero-phase filtered signal.
    """
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="high")
    return filtfilt(b, a, data)


def notch_filter(data: np.ndarray, fs: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """
    IIR notch filter to remove powerline interference.

    A narrow-band rejection filter centered at the powerline frequency
    (50 Hz in India/Europe, 60 Hz in the US). The Q factor controls
    bandwidth: higher Q = narrower notch = less distortion of nearby signal.

    Args:
        data: ECG signal.
        fs:   Sampling frequency in Hz.
        freq: Powerline frequency to reject (default 50 Hz).
        q:    Quality factor controlling notch width.

    Returns:
        Zero-phase notch-filtered signal.
    """
    b, a = iirnotch(freq, q, fs)
    return filtfilt(b, a, data)


def comb_filter(data: np.ndarray, fs: float, f0: float = 50.0) -> np.ndarray:
    """
    FIR comb filter — alternative to the IIR notch for powerline removal.

    Works by subtracting a delayed copy of the signal (one period of f0).
    Rejects f0 and ALL its harmonics simultaneously, unlike a notch which
    targets only one frequency. Useful when harmonic interference is present.

    Args:
        data: ECG signal.
        fs:   Sampling frequency in Hz.
        f0:   Fundamental interference frequency in Hz.

    Returns:
        Comb-filtered signal (causal — has slight time delay).
    """
    delay = int(fs / f0)
    b = np.zeros(delay + 1)
    b[0] = 1
    b[-1] = -1
    return lfilter(b, [1], data)


def moving_average(data: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Simple FIR moving average (boxcar) smoother.

    Acts as a low-pass filter with a sinc-shaped frequency response.
    Useful after differentiation (e.g., Pan-Tompkins pipeline) to
    smooth out the squared derivative signal before peak detection.

    Args:
        data:   Input signal.
        window: Number of samples to average.

    Returns:
        Smoothed signal (same length, zero-padded at edges via 'same').
    """
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="same")


def cascaded_moving_average(data: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Two-stage cascaded moving average.

    Convolving a boxcar with itself produces a triangular window, which has
    a much steeper frequency rolloff than a single MA (approx -40 dB/decade
    vs -20 dB/decade). Useful for stronger baseline removal without a high
    order IIR filter.

    Args:
        data:   Input signal.
        window: Window size for each stage.

    Returns:
        Double-smoothed signal.
    """
    kernel = np.ones(window) / window
    kernel2 = np.convolve(kernel, kernel)
    return np.convolve(data, kernel2, mode="same")


def derivative_filter(data: np.ndarray) -> np.ndarray:
    """
    Five-point derivative approximation (Pan-Tompkins).

    Emphasizes high slopes — exactly what happens at QRS complexes.
    After this, squaring makes all values positive and further amplifies
    large slopes relative to small ones.

    Kernel: h[n] = (1/8)(-x[n-2] - 2x[n-1] + 2x[n+1] + x[n+2])

    Args:
        data: Band-pass filtered ECG signal.

    Returns:
        Differentiated signal (same length).
    """
    kernel = np.array([1, 2, 0, -2, -1]) / 8.0
    return np.convolve(data, kernel, mode="same")


def full_pipeline(data: np.ndarray, fs: float) -> np.ndarray:
    """
    Standard ECG preprocessing pipeline in the correct order:

        Raw → Low-pass → High-pass → Notch

    This order matters:
    - LP first removes aliasing and HF noise before the HP sees the signal.
    - HP removes baseline wander.
    - Notch last is narrowest intervention — applied to an already clean signal.

    Args:
        data: Raw ECG samples.
        fs:   Sampling frequency in Hz.

    Returns:
        Preprocessed ECG signal ready for feature extraction or detection.
    """
    signal = butter_lowpass(data, cutoff=40.0, fs=fs)
    signal = butter_highpass(signal, cutoff=0.5, fs=fs)
    signal = notch_filter(signal, fs=fs, freq=50.0)
    return signal


def frequency_response(filter_func, fs: float, worN: int = 1024, **kwargs):
    """
    Utility: compute frequency response of a filter for plotting.

    Args:
        filter_func: One of the filter functions above.
        fs:          Sampling frequency in Hz.
        worN:        Number of frequency points.
        **kwargs:    Extra args forwarded to filter_func (e.g. window=20).

    Returns:
        (freqs_hz, magnitude_db) — ready to plot.
    """
    impulse = np.zeros(256)
    impulse[128] = 1.0
    response = filter_func(impulse, **kwargs)
    w, H = freqz(response, worN=worN)
    freqs_hz = w * fs / (2 * np.pi)
    magnitude_db = 20 * np.log10(np.abs(H) + 1e-12)
    return freqs_hz, magnitude_db
