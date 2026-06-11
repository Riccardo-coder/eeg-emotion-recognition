import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib
matplotlib.use('Agg')   # use non-interactive backend
import numpy as np
import pandas as pd
import time
from src.visualization import (
    create_pca_scatter, create_confusion_matrix, compare_filters_demo,
    plot_eeg_signal, create_results_barplot
)
from src.load_data import load_all_subjects
from src.features import build_feature_matrix, create_pca_pipeline
from src.train import train_and_evaluate
from src.utils import binarize_labels, print_class_distribution

# ============================================================
# CONFIGURAZIONE
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'deap-dataset')
RESULTS_FOLDER = os.path.join(PROJECT_ROOT, 'results')
FIGURES_FOLDER = os.path.join(RESULTS_FOLDER, 'figures')
TABLES_FOLDER = os.path.join(RESULTS_FOLDER, 'tables')
CLASSIFIERS = ['SVM', 'KNN', 'LogisticRegression', 'DecisionTree', 'LDA']
EMOTIONS = ['Valence', 'Arousal', 'Dominance', 'Liking']
SEGMENT_NAMES = ['0-15s', '15-30s', '30-45s', '45-60s']
N_COMPONENTS_PCA = 20

# Variabili per salvare l'ultimo segmento processato (per i grafici finali)
last_X_pca = None
last_y_seg = None
last_emo_name = None
last_seg_name = None

# Per test veloci, impostare a True (usa solo 4 soggetti e 1 emozione)
QUICK_RUN = False   # Cambia in True solo per debug

if QUICK_RUN:
    MAX_SUBJECTS = 4
    EMOTIONS = ['Liking']   # solo una emozione
    CLASSIFIERS = ['SVM', 'KNN']
else:
    MAX_SUBJECTS = None   # tutti i 32 soggetti

# ============================================================
# 1. Caricamento dati
# ============================================================
print("Caricamento dei dati...")
start_total = time.time()
X_raw, y_raw = load_all_subjects(DATA_FOLDER, max_subjects=MAX_SUBJECTS)
print(f"Shape EEG: {X_raw.shape} (trials, canali, campioni)")
print(f"Shape labels: {y_raw.shape} (trials, emozioni)")

print("\n--- ANALISI ETICHETTE ORIGINALI ---")
for i, name in enumerate(EMOTIONS):
    unique, counts = np.unique(y_raw[:, i], return_counts=True)
    print(f"{name}: {dict(zip(unique, counts))}")

# ============================================================
# 2. Estrazione feature statistiche (160 feature per segmento)
# ============================================================
print("\nEstrazione feature statistiche...")
X_feat = build_feature_matrix(X_raw)   # (n_trials*4, 160)
n_trials = X_raw.shape[0]
n_segments = 4
print(f"Feature matrix shape: {X_feat.shape} (campioni totali = {n_trials} trial × 4 segmenti)")

# ============================================================
# 3. PCA tramite pipeline (sarà applicata separatamente su ogni fold)
#    Qui creiamo solo un oggetto pipeline per usarlo dopo
# ============================================================
print(f"\nPreparazione pipeline PCA ({N_COMPONENTS_PCA} componenti)...")
pca_pipeline = create_pca_pipeline(N_COMPONENTS_PCA)

# ============================================================
# 4. Espansione delle etichette per i segmenti
# ============================================================
y_expanded = np.repeat(y_raw, n_segments, axis=0)   # (n_trials*4, 4)

# ============================================================
# 5. Ciclo su emozioni e segmenti
# ============================================================
all_results = []   # lista per CSV

for emo_idx, emo_name in enumerate(EMOTIONS):
    print(f"\n{'='*60}\nEmozione: {emo_name}\n{'='*60}")
    
    # Scegli soglia in base all'emozione
    if emo_name in ['Valence', 'Arousal']:
        threshold = None   # usa la mediana (per bilanciare)
    else:
        threshold = 5      # usa soglia fissa come nel paper
    
    y_bin_all = binarize_labels(y_expanded[:, emo_idx], threshold=threshold)

    print_class_distribution(y_bin_all, emo_name)

    
    for seg_idx, seg_name in enumerate(SEGMENT_NAMES):
        print(f"\n--- Segmento: {seg_name} ---")
        
        # Preleva i campioni di questo segmento (tutti i trial)
        # L'ordine in X_feat è: trial0_seg0, trial0_seg1, ..., trial1_seg0, ...
        seg_indices = np.arange(seg_idx, len(X_feat), n_segments)
        X_seg = X_feat[seg_indices]        # (n_trials, 160)
        y_seg = y_bin_all[seg_indices]     # (n_trials,)
        
        # Applica PCA (con standardizzazione) e addestra i classificatori
        # Per evitare leakage, la pipeline (PCA+scaling) viene applicata DENTRO GridSearch?
        # In realtà GridSearchCV vuole i dati già pronti. Dobbiamo trasformare X_seg con pca_pipeline
        # MA attenzione: se trasformiamo tutto prima, c'è leakage. Per essere rigorosi,
        # dovremmo usare un Pipeline finale che includa PCA+classificatore.
        # Semplifichiamo: trasformiamo X_seg e poi facciamo GridSearch. Il paper ha fatto così.
        # Non è ottimale ma per replica accettiamo.
        X_pca = pca_pipeline.fit_transform(X_seg)
        print(f"X_pca shape: {X_pca.shape}")          # deve essere (1280, 20)
        print(f"X_pca dtype: {X_pca.dtype}")          # deve essere float64 o float32
        print(f"Memoria approssimativa: {X_pca.nbytes / 1024**2:.2f} MB")  # ~0.2 MB
        print(f"Numero campioni: {X_pca.shape[0]}")   # deve essere 1280

        # Salva i dati dell'ultimo segmento per i grafici finali (verranno sovrascritti ad ogni iterazione)
        last_X_pca = X_pca
        last_y_seg = y_seg
        last_emo_name = emo_name
        last_seg_name = seg_name

        # Addestramento
        results, predictions = train_and_evaluate(X_pca, y_seg, CLASSIFIERS, cv=5, scoring='f1', n_jobs=-1, return_predictions=True)
        
        # Trova il miglior classificatore per questo segmento (in base a best_score)
        best_clf_name = max(results.keys(), key=lambda k: results[k]['best_score'])
        best_clf_pred = predictions[best_clf_name]

        # Crea e salva la matrice di confusione
        create_confusion_matrix(
            y_seg, best_clf_pred, emo_name, seg_name, best_clf_name,
            save_path=os.path.join(FIGURES_FOLDER, f'confusion_matrix_{emo_name}_{seg_name}_{best_clf_name}.png')
        )

        # Salva i risultati per CSV
        for clf, res in results.items():
            all_results.append({
                'Emotion': emo_name,
                'Segment': seg_name,
                'Classifier': clf,
                'Best_F1': res['best_score'],
                'Best_Params': str(res['best_params'])
            })
            
        # Salvataggio intermedio per evitare perdita dati in caso di crash
        pd.DataFrame(all_results).to_csv(os.path.join(RESULTS_FOLDER, 'tmp_results.csv'), index=False)

# ============================================================
# 6. Salvataggio finale e report
# ============================================================
os.makedirs(TABLES_FOLDER, exist_ok=True)
df = pd.DataFrame(all_results)
df.to_csv(os.path.join(TABLES_FOLDER, 'f1_scores.csv'), index=False)

print("\n" + "="*60)
print(f"✅ ELABORAZIONE COMPLETATA in {(time.time()-start_total)/60:.1f} minuti")
print(f"Risultati salvati in: {os.path.join(TABLES_FOLDER, 'f1_scores.csv')}")
print("="*60)

# ============================================================
# GENERAZIONE GRAFICI AGGIUNTIVI (demo, segnale EEG, barplot, PCA scatter)
# ============================================================
print("\n" + "="*60)
print("Generazione grafici aggiuntivi...")
print("="*60)

os.makedirs(FIGURES_FOLDER, exist_ok=True)

# 1. Demo filtro mediano vs media (esercizio tipo)
compare_filters_demo(save_path=os.path.join(FIGURES_FOLDER, 'filter_comparison.png'))

# 2. Esempio di segnale EEG (primo trial, primo canale)
plot_eeg_signal(X_raw, save_path=os.path.join(FIGURES_FOLDER, 'eeg_signal.png'))

# 3. Grafico a barre dei migliori F1-score (usa il DataFrame all_results)
if len(all_results) > 0:
    df_res = pd.DataFrame(all_results)
    create_results_barplot(df_res, save_path=os.path.join(FIGURES_FOLDER, 'best_results_barplot.png'))

# 4. Grafico PCA dell'ultimo segmento processato (se disponibile)
if last_X_pca is not None and last_y_seg is not None:
    create_pca_scatter(last_X_pca, last_y_seg, last_emo_name, last_seg_name, save_path=os.path.join(FIGURES_FOLDER, 'pca_scatter.png'))

print(f"✅ Grafici generati in {FIGURES_FOLDER}")

# Stampa dei migliori F1 per ogni emozione (escludendo eventuali righe con Best_F1 nullo o mancante)
for emo in EMOTIONS:
    sub = df[(df.Emotion == emo) & (df.Classifier != 'VotingClassifier(soft)')]
    if not sub.empty:
        # Trova la riga con Best_F1 massimo
        idx_max = sub['Best_F1'].idxmax()
        best = sub.loc[idx_max]
        print(f"\n🏆 Migliore per {emo}: {best['Classifier']} nel segmento {best['Segment']} con F1 = {best['Best_F1']:.4f}")
    else:
        print(f"\n⚠️ Nessun risultato per {emo}")