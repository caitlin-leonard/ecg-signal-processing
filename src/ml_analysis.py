"""
ml_analysis.py
--------------
Unsupervised ML on ECG beat morphology.

With only ~8 beats from a 6-second recording we can't train a supervised
classifier (no labels, too few samples). Instead we use:

1. PCA      — reduce each beat waveform (300 samples) to 2D for visualisation.
2. K-Means  — cluster beats by morphology. Different clusters could represent
              different beat types (normal, ectopic, artifact).
3. Isolation Forest — anomaly detection. Flags beats that look different from
              the majority, without needing to know what "different" means.

In a real clinical pipeline you'd have thousands of beats, labelled data (MIT-BIH),
and a trained classifier. This module demonstrates the correct approach and
shows the ML machinery works end-to-end on real ECG data.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict


def reduce_dimensions(segments: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
    """
    Project beat segments into a low-dimensional space using PCA.

    PCA finds the directions of maximum variance in the beat waveforms.
    PC1 typically captures overall amplitude shape; PC2 captures ST deviation
    or T-wave variation. Plotting beats in PC space reveals morphological clusters.

    Args:
        segments:     (N, L) array of normalised beat waveforms.
        n_components: Number of principal components to keep.

    Returns:
        reduced:  (N, n_components) projected coordinates.
        pca:      Fitted PCA object (inspect .explained_variance_ratio_).
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(segments)
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(scaled)
    return reduced, pca


def cluster_beats(
    reduced: np.ndarray,
    n_clusters: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, KMeans]:
    """
    K-Means clustering on PCA-reduced beat features.

    Groups beats by morphological similarity. In a healthy ECG all beats
    should land in a single tight cluster. Multiple clusters or a beat far
    from its cluster centre suggest ectopic beats or noise bursts.

    Args:
        reduced:      (N, k) PCA-reduced beat features.
        n_clusters:   Number of clusters to find.
        random_state: For reproducibility.

    Returns:
        labels:  (N,) cluster assignment per beat (0-indexed).
        kmeans:  Fitted KMeans object.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(reduced)
    return labels, kmeans


def detect_anomalies(
    segments: np.ndarray,
    contamination: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Isolation Forest anomaly detection on beat waveforms.

    Isolation Forest isolates anomalies by randomly partitioning the feature
    space. Anomalous points (beats with unusual morphology) require fewer
    partitions to isolate → they get lower anomaly scores.

    `contamination` is the expected fraction of anomalous beats. 0.1 means
    "assume 10% of beats are abnormal." In clinical data this would be tuned
    per patient or per rhythm type.

    Args:
        segments:      (N, L) normalised beat waveforms.
        contamination: Expected fraction of outliers in the dataset.

    Returns:
        labels:  (N,) array: +1 = normal, -1 = anomaly.
        scores:  (N,) raw anomaly scores (more negative = more anomalous).
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(segments)

    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    labels = iso.fit_predict(scaled)
    scores = iso.score_samples(scaled)
    return labels, scores


def summarise_ml_results(
    labels_kmeans: np.ndarray,
    labels_iso: np.ndarray,
    scores_iso: np.ndarray,
    rr_ms: np.ndarray,
) -> Dict:
    """
    Combine clustering and anomaly detection outputs into a readable summary.

    Args:
        labels_kmeans: K-Means cluster labels per beat.
        labels_iso:    Isolation Forest labels (+1 normal, -1 anomaly).
        scores_iso:    Isolation Forest anomaly scores.
        rr_ms:         RR intervals in milliseconds.

    Returns:
        Dictionary with summary statistics.
    """
    n_beats = len(labels_kmeans)
    n_anomalies = int(np.sum(labels_iso == -1))
    anomaly_indices = np.where(labels_iso == -1)[0]

    summary = {
        "total_beats":        n_beats,
        "n_clusters":         len(np.unique(labels_kmeans)),
        "cluster_counts":     {int(k): int(np.sum(labels_kmeans == k)) for k in np.unique(labels_kmeans)},
        "n_anomalies":        n_anomalies,
        "anomaly_fraction":   round(n_anomalies / n_beats, 3) if n_beats > 0 else 0,
        "anomaly_beat_idx":   anomaly_indices.tolist(),
        "mean_anomaly_score": float(np.mean(scores_iso[labels_iso == -1])) if n_anomalies else None,
        "mean_rr_ms":         float(np.mean(rr_ms)) if len(rr_ms) > 0 else None,
    }
    return summary
