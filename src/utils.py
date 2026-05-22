import numpy as np

def binarize_labels(y, threshold=None):
    """
    Se threshold=None, usa la mediana come soglia per bilanciare.
    """
    if threshold is None:
        threshold = np.median(y)
    return (y > threshold).astype(int)

def print_class_distribution(y_bin, emotion_name):
    """Stampa il conteggio delle classi 0 e 1."""
    unique, counts = np.unique(y_bin, return_counts=True)
    print(f"Distribuzione {emotion_name}: 0={counts[0]}, 1={counts[1]} (ratio {counts[1]/len(y_bin):.2f})")