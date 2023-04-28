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

# TPOT
from tpot import TPOTClassifier

# Utils
from utils import scale
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


def run(data_path, *args):
    # ------------------------ Read data ------------------------ #
    data = pd.read_csv(data_path)

    data.set_index('PUF_ID', inplace = True)

    X = data.drop('fpl', axis = 1)
    y = data['fpl']

    # # ------------------ Setup TPOT results file ----------------- #
    # file = open('tpot_best_config.txt', 'a')
    # file.write('TPOT Best Configurations Results:\n')
    # file.close()

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
        models = get_models(*args, class_weights = weights, seed = seed)

        # save models configurations
        if i == 0:
            file = open('models_configs.txt', 'w')

            for model_name, model in models.items():
                file.write('{}: {}\n'.format(model_name, model.get_params()))

            file.close()

        
        # --------------------- Train and Evaluate ------------------- #

        # If in the first split, initialize the results dict
        if i == 0:
            for model_name in list(models.keys()) + ['TPOT']:
                results[model_name] = {'Train Accuracy Mean': [],
                                       'Train Accuracy Median': 0,
                                       'Train Accuracy Stdev': 0,
                                       'Train F1 Mean': [],
                                       'Train F1 Median': 0,
                                       'Train F1 Stdev': 0,
                                       'Train Precision Mean': [],
                                       'Train Precision Median': 0,
                                       'Train Precision Stdev': 0,
                                       'Train Recall Mean': [],
                                        'Train Recall Median': 0,
                                       'Train Recall Stdev': 0,
                                       'Val Accuracy Mean': [],
                                       'Val Accuracy Median': 0,
                                       'Val Accuracy Stdev': 0,
                                       'Val F1 Mean': [],
                                       'Val F1 Median': 0,
                                       'Val F1 Stdev': 0,
                                       'Val Precision Mean': [],
                                       'Val Precision Median': 0,
                                       'Val Precision Stdev': 0,
                                       'Val Recall Mean': [],
                                       'Val Recall Median': 0,
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
            results[model_name]['Train Accuracy Mean'].append(accuracy_score(y_train, y_train_pred))
            results[model_name]['Train F1 Mean'].append(f1_score(y_train, y_train_pred, average = 'weighted'))
            results[model_name]['Train Precision Mean'].append(precision_score(y_train, y_train_pred, average = 'weighted'))
            results[model_name]['Train Recall Mean'].append(recall_score(y_train, y_train_pred, average = 'weighted'))

            results[model_name]['Val Accuracy Mean'].append(accuracy_score(y_val, y_val_pred))
            results[model_name]['Val F1 Mean'].append(f1_score(y_val, y_val_pred, average = 'weighted'))
            results[model_name]['Val Precision Mean'].append(precision_score(y_val, y_val_pred, average = 'weighted'))
            results[model_name]['Val Recall Mean'].append(recall_score(y_val, y_val_pred, average = 'weighted'))

    for model_name in models.keys():
        results[model_name]['Train Accuracy Stdev'] = np.std(results[model_name]['Train Accuracy Mean'])
        results[model_name]['Train Accuracy Median'] = np.median(sorted(results[model_name]['Train Accuracy Mean']))
        results[model_name]['Train Accuracy Mean'] = np.mean(results[model_name]['Train Accuracy Mean'])
        results[model_name]['Train F1 Stdev'] = np.std(results[model_name]['Train F1 Mean'])
        results[model_name]['Train F1 Median'] = np.median(sorted(results[model_name]['Train F1 Mean']))
        results[model_name]['Train F1 Mean'] = np.mean(results[model_name]['Train F1 Mean'])
        results[model_name]['Train Precision Stdev'] = np.std(results[model_name]['Train Precision Mean'])
        results[model_name]['Train Precision Median'] = np.median(sorted(results[model_name]['Train Precision Mean']))
        results[model_name]['Train Precision Mean'] = np.mean(results[model_name]['Train Precision Mean'])
        results[model_name]['Train Recall Stdev'] = np.std(results[model_name]['Train Recall Mean'])
        results[model_name]['Train Recall Median'] = np.median(sorted(results[model_name]['Train Recall Mean']))
        results[model_name]['Train Recall Mean'] = np.mean(results[model_name]['Train Recall Mean'])

        results[model_name]['Val Accuracy Stdev'] = np.std(results[model_name]['Val Accuracy Mean'])
        results[model_name]['Val Accuracy Median'] = np.median(sorted(results[model_name]['Val Accuracy Mean']))
        results[model_name]['Val Accuracy Mean'] = np.mean(results[model_name]['Val Accuracy Mean'])
        results[model_name]['Val F1 Stdev'] = np.std(results[model_name]['Val F1 Mean'])
        results[model_name]['Val F1 Median'] = np.median(sorted(results[model_name]['Val F1 Mean']))
        results[model_name]['Val F1 Mean'] = np.mean(results[model_name]['Val F1 Mean'])
        results[model_name]['Val Precision Stdev'] = np.std(results[model_name]['Val Precision Mean'])
        results[model_name]['Val Precision Median'] = np.median(sorted(results[model_name]['Val Precision Mean']))
        results[model_name]['Val Precision Mean'] = np.mean(results[model_name]['Val Precision Mean'])
        results[model_name]['Val Recall Stdev'] = np.std(results[model_name]['Val Recall Mean'])
        results[model_name]['Val Recall Median'] = np.median(sorted(results[model_name]['Val Recall Mean']))
        results[model_name]['Val Recall Mean'] = np.mean(results[model_name]['Val Recall Mean'])
        
    
    # ------------------------------------------ TPOT ----------------------------------------- #

    # ---- sample weights ---- #
    sample_weights = X['finalwt']
    X.drop('finalwt', axis = 1, inplace = True)

    # Define custom evaluation metrcis
    weighted_accuracy_scorer = make_scorer(accuracy_score)
    weighted_f1_scorer = make_scorer(f1_score, average='weighted')
    weighted_precision_scorer = make_scorer(precision_score, average='weighted')
    weighted_recall_scorer = make_scorer(recall_score, average='weighted')

    scoring = {'accuracy': weighted_accuracy_scorer,
               'f1_weighted': weighted_f1_scorer,
               'precision_weighted': weighted_precision_scorer,
               'recall_weighted': weighted_recall_scorer}
    
    # --------- Define TPOT model --------- #
    tpot = TPOTClassifier(generations = 2, population_size = 2, scoring = weighted_f1_scorer, verbosity=2, cv = skf, n_jobs=-1,
                                    random_state = seed, periodic_checkpoint_folder='/tpot_results')

    # ---- fit the model ---- #
    tpot.fit(X, y, sample_weight = sample_weights)

    # calculate cv scores
    cv_scores = cross_validate(tpot.fitted_pipeline_, X, y, cv=skf, scoring=scoring, return_train_score = True)

    results['TPOT']['Train Accuracy Mean'] = cv_scores['train_accuracy'].mean()
    results['TPOT']['Train Accuracy Median'] = np.median(sorted(cv_scores['train_accuracy']))
    results['TPOT']['Train Accuracy Stdev'] = cv_scores['train_accuracy'].std()
    results['TPOT']['Train Precision Mean'] = cv_scores['train_precision_weighted'].mean()
    results['TPOT']['Train Precision Median'] = np.median(sorted(cv_scores['train_precision_weighted']))
    results['TPOT']['Train Precision Stdev'] = cv_scores['train_precision_weighted'].std()
    results['TPOT']['Train Recall Mean'] = cv_scores['train_recall_weighted'].mean()
    results['TPOT']['Train Recall Median'] = np.median(sorted(cv_scores['train_recall_weighted']))
    results['TPOT']['Train Recall Stdev'] = cv_scores['train_recall_weighted'].std()
    results['TPOT']['Train F1 Mean'] = cv_scores['train_f1_weighted'].mean()
    results['TPOT']['Train F1 Median'] = np.median(sorted(cv_scores['train_f1_weighted']))
    results['TPOT']['Train F1 Stdev'] = cv_scores['train_f1_weighted'].std()

    results['TPOT']['Val Accuracy Mean'] = cv_scores['test_accuracy'].mean()
    results['TPOT']['Val Accuracy Median'] = np.median(sorted(cv_scores['test_accuracy']))
    results['TPOT']['Val Accuracy Stdev'] = cv_scores['test_precision_weighted'].std()
    results['TPOT']['Val Precision Mean'] = cv_scores['test_precision_weighted'].mean()
    results['TPOT']['Val Precision Median'] = np.median(sorted(cv_scores['test_precision_weighted']))
    results['TPOT']['Val Precision Stdev'] = cv_scores['test_precision_weighted'].std()
    results['TPOT']['Val Recall Mean'] = cv_scores['test_recall_weighted'].mean()
    results['TPOT']['Val Recall Median'] = np.median(sorted(cv_scores['test_recall_weighted']))
    results['TPOT']['Val Recall Stdev'] = cv_scores['test_recall_weighted'].std()
    results['TPOT']['Val F1 Mean'] = cv_scores['test_f1_weighted'].mean()
    results['TPOT']['Val F1 Median'] = np.median(sorted(cv_scores['test_f1_weighted']))
    results['TPOT']['Val F1 Stdev'] = cv_scores['test_f1_weighted'].std()

    tpot.export('tpot_pipeline.py')
    file = open('models_configs.txt', 'a')
    file.write('Best {} model\n'.format(tpot.fitted_pipeline_))
    file.close()

    return results

# Generate configurations to be tested
configs_dt = generate_configs_DT(n_models = 1)
configs_rf = generate_configs_RF(n_models = 1)
configs_gb = generate_configs_GB(n_models = 30)
configs_ab = generate_configs_AB(n_models = 15)
configs_svc = generate_configs_SVC(n_models = 21)
configs_knn = generate_configs_KNN(n_models = 16)
configs_mlp = generate_configs_MLP(n_models = 30)

start_time = time.time()
# res = run('data/preprocessed_data.csv', configs_dt, configs_rf, configs_gb, configs_ab, configs_svc, configs_knn, configs_mlp)
res = run('data/preprocessed_data.csv', configs_dt, configs_rf)

print('Time elapsed: {} seconds'.format(time.time() - start_time))

res = pd.DataFrame.from_dict(res, orient = 'index')
res.to_csv('results/results_original.csv')