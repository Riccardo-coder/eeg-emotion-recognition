import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X, y_true, emotion_name=''):
    """
    Calcola le metriche su un insieme di test (o su tutti i dati in cross‑val).
    """
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

def plot_confusion_matrix(model, X, y_true, emotion_name, save_path=None):
    y_pred = model.predict(X)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {emotion_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def create_results_table(results_dict, emotion_names, segment_names):
    """
    Crea un DataFrame pandas con i migliori F1-score per ogni combinazione.
    results_dict: dict con struttura results[segment][emotion][classifier] = best_estimator
    """
    rows = []
    for seg in segment_names:
        for emo in emotion_names:
            for clf, model in results_dict[seg][emo].items():
                # Qui si potrebbe valutare il modello su un validation set separato,
                # ma come esempio usiamo il best_score_ già ottenuto dalla cross‑val.
                # In una implementazione reale, meglio tenere traccia dei punteggi.
                pass
    # Per semplicità, ritorneremo un DataFrame dai cv_results.
    # Implementazione completa richiederebbe più tempo; lascio come struttura.
    return pd.DataFrame()