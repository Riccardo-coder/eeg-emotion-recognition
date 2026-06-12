import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def segment_trial(eeg_trial, fs=128, segment_duration=15):
    """Dividi un trial (32,8064) in 4 segmenti da 15 sec (1920 campioni)."""
    samples_per_segment = fs * segment_duration
    n_segments = eeg_trial.shape[1] // samples_per_segment
    segments = []
    for i in range(n_segments):
        start = i * samples_per_segment
        end = start + samples_per_segment
        segments.append(eeg_trial[:, start:end])
    return segments

def extract_statistical_features(segment):
    """
    Estrae 6 feature per ogni canale: media, mediana, range, varianza, deviazione standard, moda.
    Restituisce vettore di 32*6 = 192 feature.
    """
    features = []
    for ch in range(segment.shape[0]):
        sig = segment[ch, :]
        features.append(np.mean(sig))
        features.append(np.median(sig))
        features.append(np.ptp(sig))          # range = max - min
        features.append(np.var(sig))
        features.append(np.std(sig))
        features.append(mode_of_signal(sig))
    return np.array(features)

def build_feature_matrix(X_raw):
    """
    X_raw: (n_trials, 32, 8064) -> restituisce (n_trials*4, 192)
    """
    all_features = []
    for trial in range(X_raw.shape[0]):
        eeg_trial = X_raw[trial]
        segments = segment_trial(eeg_trial)
        for seg in segments:
            feat_vec = extract_statistical_features(seg)
            all_features.append(feat_vec)
    return np.array(all_features)

def create_pca_pipeline(n_components=20):
    """Crea una pipeline con StandardScaler + PCA."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components))
    ])

def mode_of_signal(signal, bins=10):
    hist, _ = np.histogram(signal, bins=bins)
    return hist.argmax()   # indice del bin più frequente