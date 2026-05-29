"""
main.py
-------
End-to-end ECG analysis pipeline.

Runs the full workflow:
  1. Load data
  2. Filter (LP → HP → Notch)
  3. Detect R-peaks and compute HRV features
  4. Segment individual beats
  5. ML: PCA + K-Means clustering + Isolation Forest anomaly detection
  6. Save all plots to outputs/

Usage:
    python main.py
    python main.py --data data/ECG2.xlsx --fs 500
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from filters import full_pipeline, moving_average, cascaded_moving_average
from features import detect_r_peaks, compute_rr_intervals, extract_hrv_features, segment_beats
from ml_analysis import reduce_dimensions, cluster_beats, detect_anomalies, summarise_ml_results

# ── constants ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
COLOURS = {
    "raw":      "#6c8ebf",
    "filtered": "#d6544b",
    "peaks":    "#2ca02c",
    "anomaly":  "#ff7f0e",
    "cluster0": "#1f77b4",
    "cluster1": "#ff7f0e",
    "cluster2": "#2ca02c",
}
CLUSTER_COLOURS = [COLOURS["cluster0"], COLOURS["cluster1"], COLOURS["cluster2"]]


def load_ecg(path: str):
    """Load ECG from Excel. Returns (time_array, ecg_array, fs)."""
    df = pd.read_excel(path)
    t   = df.iloc[:, 0].values.astype(float)
    ecg = df.iloc[:, 1].values.astype(float)
    dt  = t[1] - t[0]
    if dt <= 0:
        raise ValueError(f"Non-positive sampling interval detected: {dt}")
    fs = round(1.0 / dt)
    return t, ecg, fs


def plot_filtering_pipeline(t, ecg, filtered, peaks, save=True):
    """Figure 1: Raw vs filtered ECG with R-peaks annotated."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle("ECG Signal Preprocessing", fontsize=14, fontweight="bold")

    axes[0].plot(t, ecg, color=COLOURS["raw"], linewidth=0.9, label="Raw ECG")
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].set_title("Raw Signal")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, filtered, color=COLOURS["filtered"], linewidth=0.9, label="Filtered (LP→HP→Notch)")
    axes[1].scatter(t[peaks], filtered[peaks], color=COLOURS["peaks"], zorder=5, s=60,
                    label=f"R-peaks (n={len(peaks)})", marker="v")
    axes[1].set_ylabel("Amplitude (mV)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("After Preprocessing — R-peaks Detected")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "01_filtering_pipeline.png")
        plt.savefig(path, dpi=150)
        print(f"  Saved → {path}")
    plt.show()
    plt.close()


def plot_ma_comparison(t, ecg, save=True):
    """Figure 2: Single vs cascaded moving average smoothing + frequency response."""
    from scipy.signal import freqz

    h1 = np.ones(20) / 20
    h2 = np.convolve(h1, h1)
    w, H1 = freqz(h1, worN=1024)
    _,  H2 = freqz(h2, worN=1024)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("Moving Average Filter Analysis", fontsize=14, fontweight="bold")

    # frequency response
    axes[0].plot(w / np.pi, 20 * np.log10(np.abs(H1) + 1e-12),
                 label="Single MA (window=20)", color=COLOURS["cluster0"])
    axes[0].plot(w / np.pi, 20 * np.log10(np.abs(H2) + 1e-12),
                 label="Cascaded MA (2×20)", color=COLOURS["cluster1"], linestyle="--")
    axes[0].set_xlabel("Normalised Frequency (×π rad/sample)")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].set_title("Frequency Response\n(cascaded ≈ −40 dB/dec vs −20 dB/dec)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # time-domain smoothing effect
    single_ma    = moving_average(ecg, window=20)
    cascaded_ma  = cascaded_moving_average(ecg, window=20)
    axes[1].plot(t, ecg,         color=COLOURS["raw"],      linewidth=0.7, label="Original", alpha=0.5)
    axes[1].plot(t, single_ma,   color=COLOURS["cluster0"], linewidth=1.2, label="Single MA")
    axes[1].plot(t, cascaded_ma, color=COLOURS["cluster1"], linewidth=1.2, label="Cascaded MA", linestyle="--")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude (mV)")
    axes[1].set_title("Smoothing Effect on ECG")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "02_ma_comparison.png")
        plt.savefig(path, dpi=150)
        print(f"  Saved → {path}")
    plt.show()
    plt.close()


def plot_hrv_analysis(rr_ms, features, save=True):
    """Figure 3: RR interval tachogram and HRV feature table."""
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("Heart Rate Variability (HRV) Analysis", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    # tachogram
    ax1 = fig.add_subplot(gs[0])
    beat_numbers = np.arange(1, len(rr_ms) + 1)
    ax1.stem(beat_numbers, rr_ms, linefmt=COLOURS["filtered"],
             markerfmt="o", basefmt=" ")
    ax1.axhline(np.mean(rr_ms), color=COLOURS["peaks"], linestyle="--",
                linewidth=1.5, label=f"Mean RR = {np.mean(rr_ms):.0f} ms")
    ax1.set_xlabel("Beat Number")
    ax1.set_ylabel("RR Interval (ms)")
    ax1.set_title("RR Interval Tachogram")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # feature table
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    rows = [
        ["Mean RR",    f"{features['mean_rr_ms']:.1f} ms"],
        ["Mean HR",    f"{features['mean_hr_bpm']:.1f} bpm"],
        ["SDNN",       f"{features['sdnn_ms']:.2f} ms"],
        ["RMSSD",      f"{features['rmssd_ms']:.2f} ms"],
        ["pNN50",      f"{features['pnn50_pct']:.1f} %"],
        ["CV(RR)",     f"{features['cv_rr_pct']:.2f} %"],
    ]
    table = ax2.table(cellText=rows,
                      colLabels=["Feature", "Value"],
                      cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    ax2.set_title("Time-Domain HRV Features", pad=10)

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "03_hrv_analysis.png")
        plt.savefig(path, dpi=150)
        print(f"  Saved → {path}")
    plt.show()
    plt.close()


def plot_ml_results(t, filtered, peaks, segments, reduced, cluster_labels,
                    iso_labels, iso_scores, save=True):
    """Figure 4: PCA scatter, beat overlays by cluster, anomaly annotations."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("ML Analysis: Beat Clustering & Anomaly Detection", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # ── PCA scatter ───────────────────────────────────────────────────────────
    ax_pca = fig.add_subplot(gs[0, 0])
    unique_clusters = np.unique(cluster_labels)
    for k in unique_clusters:
        mask = cluster_labels == k
        ax_pca.scatter(reduced[mask, 0], reduced[mask, 1],
                       color=CLUSTER_COLOURS[k % len(CLUSTER_COLOURS)],
                       label=f"Cluster {k} (n={mask.sum()})", s=80, zorder=3)
    # circle anomalies
    anom_mask = iso_labels == -1
    if anom_mask.any():
        ax_pca.scatter(reduced[anom_mask, 0], reduced[anom_mask, 1],
                       facecolors="none", edgecolors=COLOURS["anomaly"],
                       s=200, linewidths=2, zorder=4, label="Anomaly (ISO Forest)")
    ax_pca.set_xlabel("PC 1")
    ax_pca.set_ylabel("PC 2")
    ax_pca.set_title("PCA of Beat Morphology")
    ax_pca.legend(fontsize=8)
    ax_pca.grid(alpha=0.3)

    # ── Beat overlays by cluster ───────────────────────────────────────────────
    time_axis = np.linspace(-200, 400, segments.shape[1])
    for k in unique_clusters:
        ax = fig.add_subplot(gs[0, k + 1])
        mask = cluster_labels == k
        for seg in segments[mask]:
            ax.plot(time_axis, seg, alpha=0.5, linewidth=0.8,
                    color=CLUSTER_COLOURS[k % len(CLUSTER_COLOURS)])
        if mask.sum() > 0:
            ax.plot(time_axis, segments[mask].mean(axis=0),
                    color="black", linewidth=2, label="Mean beat")
        ax.axvline(0, color="grey", linestyle=":", linewidth=1)
        ax.set_xlabel("Time relative to R-peak (ms)")
        ax.set_ylabel("Normalised Amplitude")
        ax.set_title(f"Cluster {k} Beats (n={mask.sum()})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # ── Full ECG with anomaly annotations ────────────────────────────────────
    ax_ecg = fig.add_subplot(gs[1, :])
    ax_ecg.plot(t, filtered, color=COLOURS["filtered"], linewidth=0.8, label="Filtered ECG")

    valid_peaks = peaks[:len(iso_labels)]   # align with segmented beats
    for i, p in enumerate(valid_peaks):
        color  = COLOURS["anomaly"] if iso_labels[i] == -1 else COLOURS["peaks"]
        marker = "x" if iso_labels[i] == -1 else "v"
        label  = "Anomalous beat" if (iso_labels[i] == -1 and i == 0) else \
                 ("Normal beat" if (iso_labels[i] == 1 and i == 0) else "_nolegend_")
        ax_ecg.scatter(t[p], filtered[p], color=color, s=80, zorder=5,
                       marker=marker, label=label)

    ax_ecg.set_xlabel("Time (s)")
    ax_ecg.set_ylabel("Amplitude (mV)")
    ax_ecg.set_title("ECG with Beat Annotations  (▼ normal  ✕ anomalous)")
    handles, labels_ = ax_ecg.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    ax_ecg.legend(by_label.values(), by_label.keys())
    ax_ecg.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "04_ml_analysis.png")
        plt.savefig(path, dpi=150)
        print(f"  Saved → {path}")
    plt.show()
    plt.close()


def print_summary(features, ml_summary):
    """Print a human-readable results summary to stdout."""
    sep = "─" * 50
    print(f"\n{sep}")
    print("  ECG ANALYSIS SUMMARY")
    print(sep)
    print(f"  Heart Rate : {features['mean_hr_bpm']:.1f} bpm")
    print(f"  Mean RR    : {features['mean_rr_ms']:.1f} ms")
    print(f"  SDNN       : {features['sdnn_ms']:.2f} ms  (overall HRV)")
    print(f"  RMSSD      : {features['rmssd_ms']:.2f} ms  (vagal tone)")
    print(f"  pNN50      : {features['pnn50_pct']:.1f}%")
    print(sep)
    print(f"  Beats segmented : {ml_summary['total_beats']}")
    print(f"  Morphology clusters : {ml_summary['n_clusters']}")
    for k, count in ml_summary['cluster_counts'].items():
        print(f"    Cluster {k}: {count} beats")
    print(f"  Anomalous beats (ISO Forest) : {ml_summary['n_anomalies']}"
          f"  ({ml_summary['anomaly_fraction']*100:.0f}%)")
    if ml_summary['anomaly_beat_idx']:
        print(f"    At beat index: {ml_summary['anomaly_beat_idx']}")
    print(sep)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ECG Signal Processing & ML Analysis")
    parser.add_argument("--data", default="../data/ECG2.xlsx", help="Path to ECG Excel file")
    parser.add_argument("--no-save", action="store_true", help="Don't save output figures")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save = not args.no_save

    # 1. Load
    print("[1/5] Loading ECG data ...")
    t, ecg, fs = load_ecg(args.data)
    print(f"      {len(ecg)} samples  |  fs = {fs} Hz  |  duration = {t[-1]:.2f} s")

    # 2. Filter
    print("[2/5] Applying preprocessing pipeline (LP → HP → Notch) ...")
    filtered = full_pipeline(ecg, fs)
    plot_filtering_pipeline(t, ecg, filtered, [], save=False)   # placeholder; peaks added below
    plot_ma_comparison(t, ecg, save=save)

    # 3. R-peaks & HRV
    print("[3/5] Detecting R-peaks and computing HRV features ...")
    peaks   = detect_r_peaks(filtered, fs)
    rr_ms   = compute_rr_intervals(peaks, fs)
    features = extract_hrv_features(rr_ms)
    plot_filtering_pipeline(t, ecg, filtered, peaks, save=save)
    plot_hrv_analysis(rr_ms, features, save=save)

    # 4. Segment beats
    print("[4/5] Segmenting individual beats ...")
    segments, valid_peaks = segment_beats(filtered, peaks, fs, pre_ms=200, post_ms=400)
    print(f"      {len(segments)} valid beat segments  |  shape: {segments.shape}")

    # 5. ML
    print("[5/5] Running ML: PCA → K-Means → Isolation Forest ...")
    n_components = min(2, len(segments) - 1)
    reduced, pca  = reduce_dimensions(segments, n_components=n_components)
    print(f"      PCA explained variance: {pca.explained_variance_ratio_.round(3)}")

    n_clusters  = min(2, len(segments))
    labels_km, _ = cluster_beats(reduced, n_clusters=n_clusters)

    contamination = min(0.4, 1.0 / len(segments) * 2)   # adaptive for short recordings
    labels_iso, scores_iso = detect_anomalies(segments, contamination=contamination)

    ml_summary = summarise_ml_results(labels_km, labels_iso, scores_iso, rr_ms)
    plot_ml_results(t, filtered, valid_peaks, segments, reduced,
                    labels_km, labels_iso, scores_iso, save=save)

    print_summary(features, ml_summary)


if __name__ == "__main__":
    main()
