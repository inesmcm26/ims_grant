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

# preprocessing:
# - fill the missing values DONE
# - no outliers
# - scaling NOT DONE

# pseudocode:
# for each split:
#   calcular sample weights
#   calcular class weights
#   scale numerical features
#   (feature selection)
#   one hot categorical variables
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

# TPOT
from tpot import TPOTClassifier

# Utils
from scalers import scale
from models import get_models, generate_configs_DT, generate_configs_RF, generate_configs_GB, generate_configs_AB, generate_configs_SVC, generate_configs_KNN, generate_configs_MLP

# Metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, make_scorer
from sklearn.model_selection import cross_validate

# Time
import time

import warnings
from sklearn.exceptions import ConvergenceWarning

# ignore convergence warnings
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# Random Seed
np.random.seed(0)
random.seed(0)
seed = 0


def run_cv(data_path, results, skf, generate_LR = False, configs_dt = None, configs_rf = None,
            configs_gb = None, configs_ab = None, configs_svm = None,
            configs_knn = None, configs_mlp = None):
    # ------------------------ Read data ------------------------ #
    data = pd.read_csv(data_path)

    data.set_index('PUF_ID', inplace = True)

    X = data.drop('fpl', axis = 1)
    y = data['fpl']

    # ------------------------ Cross-validation ------------------------ #

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
        # To train models that take this into account
        weights = {}
        class_weights = compute_class_weight('balanced', classes=[1, 2, 3], y=y_train)
        for j, value in enumerate(class_weights):
            weights[j + 1] = value # To use on model creation

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
        models = get_models(generate_LR, configs_dt , configs_rf , configs_gb , configs_ab , configs_svm , configs_knn,
               configs_mlp, class_weights = weights, seed = seed)

        # save models configurations
        if i == 0:
            file = open('results/original/models_configs.txt', 'a')

            for model_name, model in models.items():
                file.write('{}: {}\n'.format(model_name, model.get_params()))

            file.close()

        # --------------------- Train and Evaluate ------------------- #

        # If in the first split, initialize the results dict
        if i == 0:
            for model_name in list(models.keys()):
                results[model_name] = {
                    'Train Accuracy': [],
                    'Train F1': [],
                    'Train Precision': [],
                    'Train Recall': [],

                    'Val Accuracy': [],
                    'Val F1': [],
                    'Val Precision': [],
                    'Val Recall': [],
                    }
                                    
                                    
                
        for model_name, model in models.items():
            
            print('----------- TRAINING MODEL: {} -----------'.format(model_name))
            # Train
            if model_name[:3] == 'MLP':
                model.fit(X_train, y_train)
                # Evaluate
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
            elif model_name[:3] == 'KNN':
                model.fit(X_train[num_feats], y_train)
                # Evaluate
                y_train_pred = model.predict(X_train[num_feats])
                y_val_pred = model.predict(X_val[num_feats])
            else:
                model.fit(X_train, y_train, sample_weights)

                # Evaluate
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)

            # --------------------- Save Results ------------------- #
            results[model_name]['Train Accuracy'].append(accuracy_score(y_train, y_train_pred))
            results[model_name]['Train F1'].append(f1_score(y_train, y_train_pred, average = 'weighted'))
            results[model_name]['Train Precision'].append(precision_score(y_train, y_train_pred, average = 'weighted', zero_division = 1))
            results[model_name]['Train Recall'].append(recall_score(y_train, y_train_pred, average = 'weighted', zero_division = 1))

            results[model_name]['Val Accuracy'].append(accuracy_score(y_val, y_val_pred))
            results[model_name]['Val F1'].append(f1_score(y_val, y_val_pred, average = 'weighted'))
            results[model_name]['Val Precision'].append(precision_score(y_val, y_val_pred, average = 'weighted', zero_division = 1))
            results[model_name]['Val Recall'].append(recall_score(y_val, y_val_pred, average = 'weighted', zero_division = 1))

    for model_name in models.keys():
        results[model_name]['Train Accuracy Stdev'] = np.std(results[model_name]['Train Accuracy'])
        results[model_name]['Train Accuracy Median'] = np.median(sorted(results[model_name]['Train Accuracy']))
        results[model_name]['Train Accuracy Mean'] = np.mean(results[model_name]['Train Accuracy'])
        results[model_name]['Train F1 Stdev'] = np.std(results[model_name]['Train F1'])
        results[model_name]['Train F1 Median'] = np.median(sorted(results[model_name]['Train F1']))
        results[model_name]['Train F1 Mean'] = np.mean(results[model_name]['Train F1'])
        results[model_name]['Train Precision Stdev'] = np.std(results[model_name]['Train Precision'])
        results[model_name]['Train Precision Median'] = np.median(sorted(results[model_name]['Train Precision']))
        results[model_name]['Train Precision Mean'] = np.mean(results[model_name]['Train Precision'])
        results[model_name]['Train Recall Stdev'] = np.std(results[model_name]['Train Recall'])
        results[model_name]['Train Recall Median'] = np.median(sorted(results[model_name]['Train Recall']))
        results[model_name]['Train Recall Mean'] = np.mean(results[model_name]['Train Recall'])

        results[model_name]['Val Accuracy Stdev'] = np.std(results[model_name]['Val Accuracy'])
        results[model_name]['Val Accuracy Median'] = np.median(sorted(results[model_name]['Val Accuracy']))
        results[model_name]['Val Accuracy Mean'] = np.mean(results[model_name]['Val Accuracy'])
        results[model_name]['Val F1 Stdev'] = np.std(results[model_name]['Val F1'])
        results[model_name]['Val F1 Median'] = np.median(sorted(results[model_name]['Val F1']))
        results[model_name]['Val F1 Mean'] = np.mean(results[model_name]['Val F1'])
        results[model_name]['Val Precision Stdev'] = np.std(results[model_name]['Val Precision'])
        results[model_name]['Val Precision Median'] = np.median(sorted(results[model_name]['Val Precision']))
        results[model_name]['Val Precision Mean'] = np.mean(results[model_name]['Val Precision'])
        results[model_name]['Val Recall Stdev'] = np.std(results[model_name]['Val Recall'])
        results[model_name]['Val Recall Median'] = np.median(sorted(results[model_name]['Val Recall']))
        results[model_name]['Val Recall Mean'] = np.mean(results[model_name]['Val Recall'])

    return results

def run_tpot(results, data_path, skf):
    data = pd.read_csv(data_path)

    data.set_index('PUF_ID', inplace = True)

    X = data.drop('fpl', axis = 1)
    y = data['fpl']

    results['TPOT'] = {}

    # ----------------- Sample weights ----------------- #
    sample_weights = X['finalwt']
    X.drop('finalwt', axis = 1, inplace = True)

    # ----------- Custom evaluation metrics ------------ #
    weighted_accuracy_scorer = make_scorer(accuracy_score)
    weighted_f1_scorer = make_scorer(f1_score, average='weighted')
    weighted_precision_scorer = make_scorer(precision_score, average='weighted')
    weighted_recall_scorer = make_scorer(recall_score, average='weighted')

    scoring = {'accuracy': weighted_accuracy_scorer,
               'f1_weighted': weighted_f1_scorer,
               'precision_weighted': weighted_precision_scorer,
               'recall_weighted': weighted_recall_scorer}
    
    # ---------------- Define TPOT model --------------- #
    tpot = TPOTClassifier(generations = 70, population_size = 50, scoring = weighted_f1_scorer, verbosity = 2,
                            cv = skf, n_jobs=-1, random_state = seed)

    # ---- fit the model ---- #
    tpot.fit(X, y, sample_weight = sample_weights)

    # calculate cv scores
    cv_scores = cross_validate(tpot.fitted_pipeline_, X, y, cv = skf, scoring = scoring, return_train_score = True)

    results['TPOT']['Train Accuracy'] = cv_scores['train_accuracy']
    results['TPOT']['Train Accuracy Mean'] = cv_scores['train_accuracy'].mean()
    results['TPOT']['Train Accuracy Median'] = np.median(sorted(cv_scores['train_accuracy']))
    results['TPOT']['Train Accuracy Stdev'] = cv_scores['train_accuracy'].std()
    results['TPOT']['Train Precision'] = cv_scores['train_precision_weighted']
    results['TPOT']['Train Precision Mean'] = cv_scores['train_precision_weighted'].mean()
    results['TPOT']['Train Precision Median'] = np.median(sorted(cv_scores['train_precision_weighted']))
    results['TPOT']['Train Precision Stdev'] = cv_scores['train_precision_weighted'].std()
    results['TPOT']['Train Recall'] = cv_scores['train_recall_weighted']
    results['TPOT']['Train Recall Mean'] = cv_scores['train_recall_weighted'].mean()
    results['TPOT']['Train Recall Median'] = np.median(sorted(cv_scores['train_recall_weighted']))
    results['TPOT']['Train Recall Stdev'] = cv_scores['train_recall_weighted'].std()
    results['TPOT']['Train F1'] = cv_scores['train_f1_weighted']
    results['TPOT']['Train F1 Mean'] = cv_scores['train_f1_weighted'].mean()
    results['TPOT']['Train F1 Median'] = np.median(sorted(cv_scores['train_f1_weighted']))
    results['TPOT']['Train F1 Stdev'] = cv_scores['train_f1_weighted'].std()

    results['TPOT']['Val Accuracy'] = cv_scores['test_accuracy']
    results['TPOT']['Val Accuracy Mean'] = cv_scores['test_accuracy'].mean()
    results['TPOT']['Val Accuracy Median'] = np.median(sorted(cv_scores['test_accuracy']))
    results['TPOT']['Val Accuracy Stdev'] = cv_scores['test_precision_weighted'].std()
    results['TPOT']['Val Precision'] = cv_scores['test_precision_weighted']
    results['TPOT']['Val Precision Mean'] = cv_scores['test_precision_weighted'].mean()
    results['TPOT']['Val Precision Median'] = np.median(sorted(cv_scores['test_precision_weighted']))
    results['TPOT']['Val Precision Stdev'] = cv_scores['test_precision_weighted'].std()
    results['TPOT']['Val Recall'] = cv_scores['test_recall_weighted']
    results['TPOT']['Val Recall Mean'] = cv_scores['test_recall_weighted'].mean()
    results['TPOT']['Val Recall Median'] = np.median(sorted(cv_scores['test_recall_weighted']))
    results['TPOT']['Val Recall Stdev'] = cv_scores['test_recall_weighted'].std()
    results['TPOT']['Val F1'] = cv_scores['test_f1_weighted']
    results['TPOT']['Val F1 Mean'] = cv_scores['test_f1_weighted'].mean()
    results['TPOT']['Val F1 Median'] = np.median(sorted(cv_scores['test_f1_weighted']))
    results['TPOT']['Val F1 Stdev'] = cv_scores['test_f1_weighted'].std()

    tpot.export('results/original/tpot_pipeline.py')
    file = open('results/original/models_configs.txt', 'a')
    file.write('Best {} model\n'.format(tpot.fitted_pipeline_))
    file.close()

    return results
    
# Generate configurations to be tested

generate_LR = True
configs_dt = generate_configs_DT(n_models = 30)
configs_rf = generate_configs_RF(n_models = 30)
configs_gb = generate_configs_GB(n_models = 30)
configs_ab = generate_configs_AB(n_models = 15) # 15
configs_svm = generate_configs_SVC(n_models = 30) # 30
configs_knn = generate_configs_KNN(n_models = 16) # 16
configs_mlp = generate_configs_MLP(n_models = 30)

skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = seed)

start_time = time.time()

res = {}
file = open('results/original/models_configs.txt', 'w')
file.close()


# Exec 1
res = run_cv('data/data_one_hot.csv', res, skf, generate_LR = generate_LR, configs_dt = configs_dt, configs_rf = configs_rf, configs_gb = configs_gb, configs_ab = configs_ab)
res = pd.DataFrame.from_dict(res, orient = 'index')
res.to_csv('results/original/scores.csv')

# TPOT
res = pd.read_csv('results/original/scores.csv', index_col = 0).to_dict(orient = 'index')
res = run_tpot(res, 'data/data_one_hot.csv', skf)
res = pd.DataFrame.from_dict(res, orient = 'index')
res.to_csv('results/original/scores.csv')

print('Time elapsed: {} seconds'.format(time.time() - start_time))
