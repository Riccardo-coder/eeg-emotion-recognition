from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def get_classifier_and_params(name):
    """Restituisce il classificatore e la griglia di iperparametri."""
    if name == 'SVM':
        model = SVC(class_weight='balanced')
        params = {
            'C': [0.1, 1, 10],
            'gamma': [0.01, 0.1, 1],
            'kernel': ['poly', 'rbf'],
            'degree': [3]   # solo per poly
}
    elif name == 'KNN':
        model = KNeighborsClassifier()
        params = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    elif name == 'LogisticRegression':
        model = LogisticRegression(class_weight='balanced', max_iter=1000, solver='saga')
        params = {
            'C': [0.01, 0.1, 1, 10],
            'l1_ratio': [0, 0.5, 1]   # 0 = L2, 1 = L1, 0.5 = ElasticNet
        }
    elif name == 'DecisionTree':
        model = DecisionTreeClassifier(class_weight='balanced')
        params = {
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10]
        }
    elif name == 'LDA':
        model = LinearDiscriminantAnalysis()
        params = {
            'solver': ['svd', 'lsqr', 'eigen']
        }
    else:
        raise ValueError(f"Classificatore {name} sconosciuto")
    return model, params

def train_and_evaluate(X, y, classifier_names, cv=5, scoring='f1', n_jobs=-1, return_predictions=False):
    """
    Se return_predictions=True, restituisce anche le predizioni del miglior modello
    su tutti i dati (per la matrice di confusione).
    """
    results = {}
    predictions = {}
    
    for name in classifier_names:
        print(f"\n--- {name} ---")
        model, param_grid = get_classifier_and_params(name)
        grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring,
                            n_jobs=n_jobs, verbose=1)
        grid.fit(X, y)
        results[name] = {
            'best_estimator': grid.best_estimator_,
            'best_score': grid.best_score_,
            'best_params': grid.best_params_
        }
        if return_predictions:
            predictions[name] = grid.best_estimator_.predict(X)
        print(f"Miglior {scoring}: {grid.best_score_:.4f} con {grid.best_params_}")
    
    if return_predictions:
        return results, predictions
    return results