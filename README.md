
# EEG Emotion Recognition – DEAP Dataset

**Progetto di Machine Learning per il riconoscimento di emozioni da segnali EEG**  
*Corso di Interface uomo-macchina – Università degli Studi dell'Insubria*

|                     |                                   |
|---------------------|-----------------------------------|
| **Studente**        | Como Riccardo                     |
| **Matricola**       | -                                 |
| **Docente**         | Silvia Corchs                     |
| **Anno accademico** | 2025/2026                         |

---

## 📌 Abstract

Questo progetto replica il paper *"A comparative analysis of machine learning methods for emotion recognition using EEG and peripheral physiological signals"* di Doma & Pirouz (2020).  
Utilizzando il dataset **DEAP** (32 soggetti, 40 video ciascuno) e segnali **EEG** pre-processati, si classifica l’emozione in 4 dimensioni (Valenza, Arousal, Dominanza, Gradimento) mediante feature statistiche nel dominio del tempo, riduzione dimensionale con PCA e classificatori classici (SVM, KNN, Logistic Regression, Decision Tree, LDA). È stato inoltre implementato un **VotingClassifier** (soft voting) per combinare i modelli migliori.

**Risultati migliori ottenuti (F1-score):**  
- **Liking**: 0.799 (SVM, kernel RBF)  
- **Dominance**: 0.755 (SVM, kernel RBF)  
- **Valence**: 0.667 (SVM, kernel polinomiale)  
- **Arousal**: 0.654 (SVM, kernel polinomiale/RBF)

---

## 📁 Struttura del repository

```
eeg-emotion-recognition/
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── load_data.py         # Caricamento file .dat
│   ├── features.py          # Segmentazione, statistiche, PCA
│   ├── train.py             # Classificatori e GridSearch
│   ├── evaluate.py          # Metriche, confusion matrix
│   ├── utils.py             # Utility (binarizzazione, salvataggio)
│   └── visualization.py     # Funzioni per grafici (PCA, confusion matrix, barplot)
├── scripts/
│   └── run_pipeline.py      # Script principale (esegue l'intera pipeline)
└── results/                 # Generato automaticamente
    ├── tables/              # CSV con i punteggi F1
    └── figures/             # Grafici (PCA, confusion matrix, barplot, segnale EEG)
```

---

## ⚙️ Requisiti

- Python 3.9 o superiore
- Pip e ambiente virtuale (consigliato)

Installa le dipendenze:

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
numpy
pandas
scikit-learn
scipy
matplotlib
seaborn
joblib
```

---

## 🧠 Dataset

Viene utilizzato il dataset **DEAP** (Database for Emotion Analysis using Physiological Signals), scaricato da:  
[https://www.kaggle.com/datasets/manh123df/deap-dataset](https://www.kaggle.com/datasets/manh123df/deap-dataset)

I file pre-processati (`.dat`) contengono un array `data` di forma `(40, 40, 8064)`, dove:
- **40** trial (video) per soggetto.
- **40** canali fisiologici totali (32 EEG + 8 periferici).
- **8064** campioni temporali (60 secondi a 128 Hz + baseline).

In questo progetto vengono utilizzati **solo i 32 canali EEG**, come nel paper di riferimento.

L'array `labels` ha forma `(40, 4)` e contiene le valutazioni soggettive per Valenza, Arousal, Dominanza, Gradimento (scale 1-9).  
Le etichette sono binarizzate con soglia `>5` (classe positiva se >5).

Posizionare i file `.dat` (da `s01.dat` a `s32.dat`) nella cartella:  
`data/deap_dataset/data_preprocessed_python/`

**Riferimento originale:**  
Koelstra, S., Muhl, C., Soleymani, M., et al. (2011). *DEAP: A database for emotion analysis; using physiological signals*. IEEE Transactions on Affective Computing, 3(1), 18-31.

---

## 🚀 Esecuzione

1. Clona il repository:
   ```bash
   git clone https://github.com/Riccardo-coder/eeg-emotion-recognition.git
   cd eeg-emotion-recognition
   ```

2. Crea e attiva l’ambiente virtuale:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   venv\Scripts\activate           # Windows
   ```

3. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```

4. Posiziona i file `.dat` nella cartella:
   ```
   data/deap_dataset/data_preprocessed_python/s01.dat
   ```

5. Lancia lo script principale:
   ```bash
   python scripts/run_pipeline.py
   ```

Al termine, i risultati vengono salvati in `results/tables/f1_scores.csv` e i grafici in `results/figures/`.

---

## 📊 Risultati ottenuti (confronto con il paper)

| Emozione   | Miglior classificatore | Segmento | F1-score (nostro) | F1-score (paper) |
|------------|------------------------|----------|-------------------|------------------|
| Valence    | SVM (poly)             | 45‑60s   | 0.667             | ~0.78            |
| Arousal    | SVM (poly/rbf)         | 45‑60s   | 0.654             | ~0.85            |
| Dominance  | SVM (rbf)              | 45‑60s   | 0.755             | ~0.83            |
| Liking     | SVM (rbf)              | 0‑15s/30‑45s | 0.799         | ~0.84            |

Le differenze sono dovute principalmente all’uso di kernel lineare per ragioni computazionali (nel paper vengono usati kernel polinomiali/RBF su tutti i classificatori) e alla versione del dataset scaricata da Kaggle (possibile diversa distribuzione delle etichette).

---

## 📝 Note implementative

- **Segmentazione:** ogni trial di 60 secondi è diviso in 4 intervalli da 15 secondi (1920 campioni a 128 Hz).
- **Feature estratte:** per ogni canale EEG → media, mediana, range, varianza, deviazione standard (5 feature per canale → 160 feature totali per segmento).
- **PCA:** riduzione a 20 componenti con standardizzazione (spiega circa l’85% della varianza).
- **Classificatori:** SVM, k‑NN, Logistic Regression, Decision Tree, LDA, VotingClassifier (soft voting).
- **Ottimizzazione:** GridSearchCV con 5‑fold cross‑validation, metrica `f1`.
- **Bilanciamento classi:** `class_weight='balanced'` per SVM, LogisticRegression e DecisionTree per mitigare lo squilibrio (Valence/Arousal avevano solo ~21% positivi con soglia 5).
- **Generazione grafici:** matrice di confusione, PCA scatter, segnale EEG, barplot dei migliori F1, demo filtri (media vs mediano).

---

## 🖼️ Grafici generati

Lo script produce automaticamente i seguenti grafici nella cartella `results/figures/`:

- `confusion_matrix_*.png` – matrici di confusione per ogni emozione/segmento/miglior classificatore.
- `pca_scatter.png` – distribuzione dei campioni dopo PCA (blu=negativo, rosso=positivo).
- `eeg_signal.png` – esempio di segnale EEG reale (primo trial, primo canale).
- `best_results_barplot.png` – barplot dei migliori F1 per emozione.
- `filter_comparison.png` – demo didattica filtro mediano vs filtro media (esercizio tipo).

---

## 👤 Autore

- **Como Riccardo** – [N. matricola]  
- **Corso di laurea:** Informatica  
- **Email:** rcomo1@studenti.uninsubria.it  
- **GitHub:** [https://github.com/Riccardo-coder/](https://github.com/Riccardo-coder/)

---

## 📚 Riferimenti

- Doma, V., Pirouz, M. (2020). *A comparative analysis of machine learning methods for emotion recognition using EEG and peripheral physiological signals*. Journal of Big Data, 7, 18.  
- Koelstra, S., et al. (2011). *DEAP: A database for emotion analysis using physiological signals*. IEEE Transactions on Affective Computing, 3(1), 18‑31.

---

## 📄 Licenza

Questo progetto è realizzato esclusivamente per fini didattici. Il dataset DEAP è soggetto alla propria licenza (richiedere il permesso agli autori).