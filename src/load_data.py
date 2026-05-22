import pickle
import glob
import os
import numpy as np

def load_all_subjects(data_folder, max_subjects=None):
    """
    Carica i file .dat dalla cartella che contiene s01.dat ... s32.dat.
    """
    X_list = []
    y_list = []
    file_pattern = os.path.join(data_folder, 's*.dat')
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        raise FileNotFoundError(f"Nessun file .dat trovato in {data_folder}")
    
    if max_subjects is not None:
        files = files[:max_subjects]
    
    for fpath in files:
        print(f"Caricamento {fpath}...")
        with open(fpath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        eeg = data['data'][:, :32, :]   # (40, 32, 8064)
        labels = data['labels']          # (40, 4)
        X_list.append(eeg)
        y_list.append(labels)
    
    # Concatena lungo la prima dimensione
    X = np.concatenate(X_list, axis=0)   # (n_soggetti*40, 32, 8064)
    y = np.concatenate(y_list, axis=0)
    return X, y