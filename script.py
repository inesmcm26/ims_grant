# models:
# - Logistic Regression
# - Random Forest
# - Gradient Boosting
# - Adaptive Boosting
# - SVM
# - KNN
# - Naive Bayes
# - Decision Tree (?)
# - MLP
# - TPOT

# metrics:
# f1 score averaged
# precision averaged
# recall averaged
# ROC AUC averaged

# preprocessing:
# - fill the missing values DONE
# - no outliers
# - scaling NOT DONE

# pseudocode:
# for each split:
#   missing values(?)
#   scale numerical features
#   (feature selection)
#   calcular class weights
#   for each algoritmo (e set de parâmetros):
#       treinar em X_train, y_train (com sample weights)
#       testar em y_train_pred, y_train (com class weights)
#       testar em y_val_pred, y_val (com class weights)
#       adicionar a dict_performance
# calcular média de performances de cada algoritmo

# dict_performance: {'algoritmo X': {f1_train : [], f1_val: [], p_train: [], p_val: [] etc.},
#                    'algoritmo Y': {f1_train : [], f1_val: [], p_train: [], p_val: [] etc.}
#                   }

# dict_algoritmos: {'algoritmo X': objecto X,
#                   'algoritmo Y': objecto Y}

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import random
import numpy as np
import pandas as pd

from preprocess import scale

np.random.seed(0)
random.seed(0)
seed = 0

def run(data_path):
    # ------------------------ Read data ------------------------ #
    data = pd.read_csv(data_path)

    X = data.drop('fpl', axis = 1)
    y = data['fpl']

    # ------------------------ Cross-validation ------------------------ #

    skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = seed)

    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # -------------------- Sample Weights --------------------- #

        # To use when training the model
        sample_weights = X_train['finalwt']
        X_train.drop('finalwt', axis = 1, inplace = True)
        X_val.drop('finalwt', axis = 1, inplace = True)

        # --------------------- Class Weights --------------------- #
        weights = {}
        class_weights = compute_class_weight('balanced', classes=[1, 2, 3], y=y_train)
        for i, value in enumerate(class_weights):
            weights[i + 1] = value

        # To use in model evaluation
        X_train_weights = y_train.map({k: v for k, v in weights.items()})
        X_val_weights = y_val.map({k: v for k, v in weights.items()})

        # -------------- Split numerical and nominal -------------- #
        X_train_num = X_train[list(filter(lambda x: X_train[x].nunique() > 2, X_train.columns.values))]
        X_train_cat = X_train[list(filter(lambda x: X_train[x].nunique() == 2, X_train.columns.values))]

        X_val_num = X_val[list(filter(lambda x: X_val[x].nunique() > 2, X_val.columns.values))]
        X_val_cat = X_val[list(filter(lambda x: X_val[x].nunique() == 2, X_val.columns.values))]

        # ------------------------ Scaling ------------------------ #
        X_train_num, X_val_num = scale(X_train_num, X_val_num)

        

