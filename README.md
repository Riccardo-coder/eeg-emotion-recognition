
# EEG Emotion Recognition – DEAP Dataset

**Progetto di Machine Learning per il riconoscimento di emozioni da segnali EEG**  
*Corso di Interface uomo-macchina – Università degli Studi dell'Insubria*

|                     |                                   |
|---------------------|-----------------------------------|
| **Studente**        | Como Riccardo                     |
| **Matricola**       | 758697                            |
| **Docente**         | Silvia Corchs                     |
| **Anno accademico** | 2025/2026                         |

---

## 📌 Abstract

Questo progetto replica il paper *"A comparative analysis of machine learning methods for emotion recognition using EEG and peripheral physiological signals"* di Doma & Pirouz (2020).  
Utilizzando il dataset **DEAP** (32 soggetti, 40 video ciascuno) e segnali **EEG** pre‑processati, si classifica l’emozione in 4 dimensioni (Valenza, Arousal, Dominanza, Gradimento) mediante feature statistiche nel dominio del tempo, riduzione dimensionale con PCA e classificatori classici (SVM, KNN, Logistic Regression, Decision Tree, LDA). È stato inoltre implementato un **VotingClassifier** (soft voting) per combinare i modelli migliori.

**Risultati migliori ottenuti (F1‑score):**  
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
│   ├── load_data.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   └── visualization.py
├── scripts/
│   └── run_pipeline.py
└── data/
    └── deap-dataset/            # cartella che contiene i file .dat
        ├── s01.dat
        ├── s02.dat
        └── ... (fino a s32.dat)
```

> **Nota:** La cartella `data/` non è inclusa nel repository perché i file `.dat` sono grandi (~3 GB) e soggetti a licenza. Dovrai crearla manualmente.

---

## ⚙️ Requisiti

- Python 3.9+
- Almeno 8 GB di RAM
- ~3 GB di spazio libero

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

## 🧠 Dataset – DEAP

Scarica il dataset da:  
[https://www.kaggle.com/datasets/manh123df/deap-dataset](https://www.kaggle.com/datasets/manh123df/deap-dataset)

Copia i 32 file `.dat` (da `s01.dat` a `s32.dat`) nella cartella `data/deap-dataset/` (creala se non esiste).  
Il percorso finale deve essere ad esempio: `data/deap-dataset/s01.dat`

**Non rinominare la cartella** – il codice cerca esattamente `DATA_FOLDER = 'data/deap-dataset/'`.

---

## 🚀 Esecuzione

1. **Clona il repository**

   ```bash
   git clone https://github.com/Riccardo-coder/eeg-emotion-recognition.git
   cd eeg-emotion-recognition
   ```

2. **Crea e attiva l'ambiente virtuale**

   Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   Linux/Mac:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Installa le dipendenze**

   ```bash
   pip install -r requirements.txt
   ```

4. **Posiziona i file `.dat`** nella cartella `data/deap-dataset/`.

5. **(Opzionale) Abilita la modalità veloce per test**  
   Apri `scripts/run_pipeline.py` e imposta `QUICK_RUN = True`.

6. **Lancia lo script**

   ```bash
   python scripts/run_pipeline.py
   ```

7. **Risultati**  
   - CSV: `results/tables/f1_scores.csv`  
   - Grafici: `results/figures/` (matrici di confusione, PCA scatter, barplot, segnale EEG, demo filtri)

---

## 📊 Risultati ottenuti (confronto con il paper)

| Emozione   | Miglior classificatore | Segmento | F1-score (nostro) | F1-score (paper) |
|------------|------------------------|----------|-------------------|------------------|
| Valence    | SVM (poly)             | 45‑60s   | 0.667             | ~0.78            |
| Arousal    | SVM (poly/rbf)         | 45‑60s   | 0.654             | ~0.85            |
| Dominance  | SVM (rbf)              | 45‑60s   | 0.755             | ~0.83            |
| Liking     | SVM (rbf)              | 0‑15s / 30‑45s | 0.799     | ~0.84            |

Le differenze sono dovute principalmente all’uso di kernel lineare in alcuni casi (per ragioni computazionali) e alla versione del dataset.

---

## 📝 Note implementative

- Segmentazione in 4 intervalli da 15 secondi
- Feature: media, mediana, range, varianza, std (160 per segmento)
- PCA a 20 componenti
- Classificatori: SVM, k‑NN, Logistic Regression, Decision Tree, LDA, VotingClassifier
- Ottimizzazione: GridSearchCV (5‑fold, metrica F1)
- Bilanciamento: `class_weight='balanced'`
- Grafici generati automaticamente

---

## 👤 Autore

- **Como Riccardo**  
- **Email:** rcomo1@studenti.uninsubria.it  
- **GitHub:** [https://github.com/Riccardo-coder/](https://github.com/Riccardo-coder/)

---

## 📚 Riferimenti

- Doma, V., Pirouz, M. (2020). *A comparative analysis of machine learning methods for emotion recognition using EEG and peripheral physiological signals*. Journal of Big Data, 7, 18.  
- Koelstra, S., et al. (2011). *DEAP: A database for emotion analysis using physiological signals*. IEEE Transactions on Affective Computing, 3(1), 18‑31.

---

## 📄 Licenza

Progetto didattico. Il dataset DEAP ha licenza propria.
