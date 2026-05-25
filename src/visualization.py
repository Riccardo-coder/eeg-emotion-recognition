import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.signal import medfilt
import os

def create_pca_scatter(X_pca, y_seg, emotion_name, segment_name, save_path=None):
    """
    Grafico a dispersione delle prime 2 componenti PCA.
    """
    plt.figure(figsize=(10, 8))
    colors = ['red' if y == 1 else 'blue' for y in y_seg]
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, s=30)
    plt.xlabel('PC1', fontsize=12)
    plt.ylabel('PC2', fontsize=12)
    plt.title(f'PCA - {emotion_name} - {segment_name}\n(Blu=Negativo, Rosso=Positivo)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Grafico PCA salvato: {save_path}")
    plt.close()

def create_confusion_matrix(y_true, y_pred, emotion_name, segment_name, classifier_name, save_path=None):
    """
    Matrice di confusione per un classificatore.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negativo (≤5)', 'Positivo (>5)'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Matrice di Confusione - {emotion_name} - {segment_name}\n{classifier_name}', fontsize=12)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Matrice confusione salvata: {save_path}")
    plt.close()

def compare_filters_demo(save_path=None):
    """
    Dimostrazione filtro mediano vs filtro media (come esercizio 7.3 del corso).
    """
    # Segnale con spike come nell'esempio del corso
    x = np.array([2, 3, 2, 20, 3, 2, 1], dtype=float)
    
    # Filtro mediano (finestra 3)
    x_median = medfilt(x, kernel_size=3)
    
    # Filtro media (finestra 3)
    def moving_average(signal, window=3):
        result = np.copy(signal)
        half = window // 2
        for i in range(half, len(signal) - half):
            result[i] = np.mean(signal[i-half:i+half+1])
        return result
    
    x_mean = moving_average(x)
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, 'o-', label='Segnale Originale', linewidth=2, markersize=8, color='black')
    plt.plot(x_median, 's-', label='Filtro Mediano (W=3)', linewidth=2, markersize=8, color='green')
    plt.plot(x_mean, '^-', label='Filtro Media (W=3)', linewidth=2, markersize=8, color='red')
    plt.legend(fontsize=12)
    plt.xlabel('Campione', fontsize=12)
    plt.ylabel('Ampiezza', fontsize=12)
    plt.title('Confronto Filtro Mediano vs Filtro Media', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Evidenzia lo spike originale
    plt.annotate('Spike impulsivo', xy=(3, 20), xytext=(4, 25),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Demo filtri salvata: {save_path}")
    plt.close()

def plot_eeg_signal(X_raw, save_path=None):
    """
    Grafico di un canale EEG di esempio.
    """
    # Primo trial, primo canale, primi 1000 campioni
    signal = X_raw[0, 0, :1000]
    
    plt.figure(figsize=(12, 4))
    plt.plot(signal, linewidth=0.8, color='blue')
    plt.xlabel('Campioni (128 Hz)', fontsize=12)
    plt.ylabel('Ampiezza (µV)', fontsize=12)
    plt.title('Esempio di segnale EEG (canale F3, primi 1000 campioni)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Grafico EEG salvato: {save_path}")
    plt.close()

def create_results_barplot(results_df, save_path=None):
    """
    Grafico a barre dei migliori F1 per emozione.
    """
    emotions = results_df['Emotion'].unique()
    best_f1 = []
    classifiers = []
    
    for emo in emotions:
        sub = results_df[results_df['Emotion'] == emo]
        best_row = sub.loc[sub['Best_F1'].idxmax()]
        best_f1.append(best_row['Best_F1'])
        classifiers.append(best_row['Classifier'])
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(emotions, best_f1, color=colors, edgecolor='black', linewidth=1.5)
    plt.ylim(0, 1.0)
    plt.ylabel('Miglior F1-score', fontsize=12)
    plt.xlabel('Emozione', fontsize=12)
    plt.title('Migliori risultati per emozione', fontsize=14)
    
    # Aggiungi etichette con i valori e il classificatore
    for bar, f1, clf in zip(bars, best_f1, classifiers):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{f1:.3f}\n({clf})', ha='center', va='bottom', fontsize=9)
    
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Barplot salvato: {save_path}")
    plt.close()