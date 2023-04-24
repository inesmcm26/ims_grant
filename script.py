# models:
# - Logistic Regression
# - Random Forest
# - Gradient Boosting
# - Adaptive Boosting
# - SVM
# - KNN
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
#   calcular sample weights
#   calcular class weights
#   missing values(?)
#   scale numerical features
#   (feature selection)
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

# Utils
from utils import scale
from models import get_models, generate_configs_DT, generate_configs_RF, generate_configs_GB, generate_configs_AB, generate_configs_SVC, generate_configs_KNN, generate_configs_MLP

# Metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Time
import time

# Random Seed
np.random.seed(0)
random.seed(0)
seed = 0

# TODO: queremos um set de teste à parte para testar o melhor algoritmo no final em separado?


def run(data_path, *args):
    # ------------------------ Read data ------------------------ #
    data = pd.read_csv(data_path)

    data.set_index('PUF_ID', inplace = True)

    X = data.drop('fpl', axis = 1)
    y = data['fpl']

    # ------------------------ Cross-validation ------------------------ #
    results = {}

    skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = seed)

    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):

        print('-------------------- Split {} -----------------'.format(i + 1))

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # -------------------- Sample Weights --------------------- #

        # To use when training the model
        sample_weights = X_train['finalwt']
        X_train = X_train.drop('finalwt', axis = 1)
        X_val = X_val.drop('finalwt', axis = 1)

        # --------------------- Class Weights --------------------- #
        weights = {}
        class_weights = compute_class_weight('balanced', classes=[1, 2, 3], y=y_train)
        for j, value in enumerate(class_weights):
            weights[j + 1] = value # To use on model creation

        # To use in model evaluation
        X_train_weights = y_train.map({k: v for k, v in weights.items()})
        X_val_weights = y_val.map({k: v for k, v in weights.items()})

        # -------------- Split numerical and nominal -------------- #
        
        cat_feats = list(X_train.columns[X_train.nunique() == 2])
        num_feats = list(X_train.columns[X_train.nunique() > 2])


        X_train_num = X_train[num_feats]
        X_train_cat = X_train[cat_feats]

        X_val_num = X_val[num_feats]
        X_val_cat = X_val[cat_feats]

        # ------------------------ Scaling ------------------------ #
        X_train_num, X_val_num = scale(X_train_num, X_val_num)

        # --------------------- Concatenate ------------------- #
        X_train = pd.concat([X_train_num, X_train_cat], axis = 1)
        X_val = pd.concat([X_val_num, X_val_cat], axis = 1)

        # --------------------- Generate Models ------------------- #
        models = get_models(*args, class_weights = weights, seed = seed)

        # --------------------- Train and Evaluate ------------------- #

        # If in the first split, initialize the results dict
        if i == 0:
            for model_name in models.keys():
                results[model_name] = {'Train Accuracy': [],
                                       'Train Accuracy Stdev': 0,
                                       'Train F1': [],
                                       'Train F1 Stdev': 0,
                                       'Train Precision': [],
                                       'Train Precision Stdev': 0,
                                       'Train Recall': [],
                                       'Train Recall Stdev': 0,
                                       'Val Accuracy': [],
                                       'Val Accuracy Stdev': 0,
                                       'Val F1': [],
                                       'Val F1 Stdev': 0,
                                       'Val Precision': [],
                                       'Val Precision Stdev': 0,
                                       'Val Recall': [],
                                       'Val Recall Stdev': 0}
                
                
        
        for model_name, model in models.items():
            

            # Train
            if model_name[:3] in ['KNN', 'MLP']:
                print('----------- TRAINING MODEL: {} -----------'.format(model_name))
                # print(model)
                # TODO: Quando é KNN usamos só as features numéricas para calculatr os KNN ? 
                model.fit(X_train, y_train)
            else:
                print('----------- TRAINING MODEL: {} -----------'.format(model_name))
                # print(model)
                model.fit(X_train, y_train, sample_weights)

            # Evaluate
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            # --------------------- Save Results ------------------- #
            results[model_name]['Train Accuracy'].append(accuracy_score(y_train, y_train_pred, sample_weight = X_train_weights))
            results[model_name]['Train F1'].append(f1_score(y_train, y_train_pred, average = 'weighted', sample_weight = X_train_weights))
            results[model_name]['Train Precision'].append(precision_score(y_train, y_train_pred, average = 'weighted', sample_weight = X_train_weights))
            results[model_name]['Train Recall'].append(recall_score(y_train, y_train_pred, average = 'weighted', sample_weight = X_train_weights))

            results[model_name]['Val Accuracy'].append(accuracy_score(y_val, y_val_pred, sample_weight = X_val_weights))
            results[model_name]['Val F1'].append(f1_score(y_val, y_val_pred, average = 'weighted', sample_weight = X_val_weights))
            results[model_name]['Val Precision'].append(precision_score(y_val, y_val_pred, average = 'weighted', sample_weight = X_val_weights))
            results[model_name]['Val Recall'].append(recall_score(y_val, y_val_pred, average = 'weighted', sample_weight = X_val_weights))


    for model_name in models.keys():
        results[model_name]['Train Accuracy Stdev'] = np.std(results[model_name]['Train Accuracy'])
        results[model_name]['Train Accuracy'] = np.mean(results[model_name]['Train Accuracy'])
        results[model_name]['Train F1 Stdev'] = np.std(results[model_name]['Train F1'])
        results[model_name]['Train F1'] = np.mean(results[model_name]['Train F1'])
        results[model_name]['Train Precision Stdev'] = np.std(results[model_name]['Train Precision'])
        results[model_name]['Train Precision'] = np.mean(results[model_name]['Train Precision'])
        results[model_name]['Train Recall Stdev'] = np.std(results[model_name]['Train Recall'])
        results[model_name]['Train Recall'] = np.mean(results[model_name]['Train Recall'])

        results[model_name]['Val Accuracy Stdev'] = np.std(results[model_name]['Val Accuracy'])
        results[model_name]['Val Accuracy'] = np.mean(results[model_name]['Val Accuracy'])
        results[model_name]['Val F1 Stdev'] = np.std(results[model_name]['Val F1'])
        results[model_name]['Val F1'] = np.mean(results[model_name]['Val F1'])
        results[model_name]['Val Precision Stdev'] = np.std(results[model_name]['Val Precision'])
        results[model_name]['Val Precision'] = np.mean(results[model_name]['Val Precision'])
        results[model_name]['Val Recall Stdev'] = np.std(results[model_name]['Val Recall'])
        results[model_name]['Val Recall'] = np.mean(results[model_name]['Val Recall'])
        
    return results

# Generate configurations to be tested
configs_dt = generate_configs_DT(n_models = 5)
configs_rf = generate_configs_RF(n_models = 3)
configs_gb = generate_configs_GB(n_models = 150)
configs_ab = generate_configs_AB(n_models = 15)
configs_svc = generate_configs_SVC(n_models = 21)
configs_knn = generate_configs_KNN(n_models = 16)
configs_mlp = generate_configs_MLP(n_models = 150)

start_time = time.time()
# res = run('data/preprocessed_data.csv', configs_dt, configs_rf, configs_gb, configs_ab, configs_svc, configs_knn, configs_mlp)
res = run('data/preprocessed_data.csv', configs_dt, configs_rf)

print('Time elapsed: {} seconds'.format(time.time() - start_time))

res = pd.DataFrame.from_dict(res, orient = 'index')
res.to_csv('results/preprocessed_data_results.csv')