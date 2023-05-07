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

# Feature Selection
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

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


def run_cv(data_path, results, skf, min_feats = 80, tol = 0.01, generate_LR = False, configs_dt = None, configs_rf = None,
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

        # ------------------- Feature Selection -------------------- #
                
        # Selects the K most important features according to RFE.
        # The k is dinamically choosen by calculating the scores of the model with values of K
        # ranging from min_feats to the number of features in the dataset.
        nr_features_list = list(range(min_feats, len(X_train.columns) +1))
        highest_score = 0

        # to select the best nr of features to keep
        for n in nr_features_list:
            # create model for the RFE to use
            model = DecisionTreeClassifier(random_state = seed)

            # create RFE object to select the best n features
            rfe = RFE(estimator = model, n_features_to_select = n)
            X_train_rfe = rfe.fit_transform(X_train, y_train, sample_weight = sample_weights)
            X_val_rfe = rfe.transform(X_val)

            # fit the model with the selected features
            model.fit(X_train_rfe, y_train, sample_weight = sample_weights)
            
            # model score
            y_pred = model.predict(X_val_rfe)
            score = f1_score(y_val, y_pred, average = 'weighted')
            print(score)

            # select the number of features that show the highest score
            if(score > (highest_score + tol)):
                selected_features = X_train.columns[rfe.support_].values
                highest_score = score

        # Save selected features in each split
        file = open('results/feat_sel/selected_features.txt', 'a')
        file.write('Split {} | Nr of selected features: {}\n'.format(i + 1, len(selected_features)))
        file.write('Features: {}\n'.format(selected_features))
        file.write('-----------------------------------------------------------')
        file.close()
    

    return 0

    #     # -------------------- One hot encoding ------------------- #
    #     # TODO

    #     # -------------- Split numerical and nominal -------------- #
        
    #     cat_feats = list(X_train.columns[X_train.nunique() == 2])
    #     num_feats = list(X_train.columns[X_train.nunique() > 2])


    #     X_train_num = X_train[num_feats]
    #     X_train_cat = X_train[cat_feats]

    #     X_val_num = X_val[num_feats]
    #     X_val_cat = X_val[cat_feats]

    #     # ------------------------ Scaling ------------------------ #
    #     X_train_num, X_val_num = scale(X_train_num, X_val_num)

    #     # --------------------- Concatenate ------------------- #
    #     X_train = pd.concat([X_train_num, X_train_cat], axis = 1)
    #     X_val = pd.concat([X_val_num, X_val_cat], axis = 1)

    #     # --------------------- Generate Models ------------------- #
    #     models = get_models(generate_LR, configs_dt , configs_rf , configs_gb , configs_ab , configs_svm , configs_knn,
    #            configs_mlp, class_weights = weights, seed = seed)

    #     # save models configurations
    #     if i == 0:
    #         file = open('results/feat_sel/models_configs.txt', 'a')

    #         for model_name, model in models.items():
    #             file.write('{}: {}\n'.format(model_name, model.get_params()))

    #         file.close()

    #     # --------------------- Train and Evaluate ------------------- #

    #     # If in the first split, initialize the results dict
    #     if i == 0:
    #         for model_name in list(models.keys()):
    #             results[model_name] = {
    #                 'Train Accuracy': [],
    #                 'Train F1': [],
    #                 'Train Precision': [],
    #                 'Train Recall': [],

    #                 'Val Accuracy': [],
    #                 'Val F1': [],
    #                 'Val Precision': [],
    #                 'Val Recall': [],
    #                 }
                                    
                                    
                
    #     for model_name, model in models.items():
            
    #         print('----------- TRAINING MODEL: {} -----------'.format(model_name))
    #         # Train
    #         if model_name[:3] == 'MLP':
    #             model.fit(X_train, y_train)
    #             # Evaluate
    #             y_train_pred = model.predict(X_train)
    #             y_val_pred = model.predict(X_val)
    #         elif model_name[:3] == 'KNN':
    #             model.fit(X_train[num_feats], y_train)
    #             # Evaluate
    #             y_train_pred = model.predict(X_train[num_feats])
    #             y_val_pred = model.predict(X_val[num_feats])
    #         else:
    #             model.fit(X_train, y_train, sample_weights)

    #             # Evaluate
    #             y_train_pred = model.predict(X_train)
    #             y_val_pred = model.predict(X_val)

    #         # --------------------- Save Results ------------------- #
    #         results[model_name]['Train Accuracy'].append(accuracy_score(y_train, y_train_pred))
    #         results[model_name]['Train F1'].append(f1_score(y_train, y_train_pred, average = 'weighted'))
    #         results[model_name]['Train Precision'].append(precision_score(y_train, y_train_pred, average = 'weighted', zero_division = 1))
    #         results[model_name]['Train Recall'].append(recall_score(y_train, y_train_pred, average = 'weighted', zero_division = 1))

    #         results[model_name]['Val Accuracy'].append(accuracy_score(y_val, y_val_pred))
    #         results[model_name]['Val F1'].append(f1_score(y_val, y_val_pred, average = 'weighted'))
    #         results[model_name]['Val Precision'].append(precision_score(y_val, y_val_pred, average = 'weighted', zero_division = 1))
    #         results[model_name]['Val Recall'].append(recall_score(y_val, y_val_pred, average = 'weighted', zero_division = 1))

    # for model_name in models.keys():
    #     results[model_name]['Train Accuracy Stdev'] = np.std(results[model_name]['Train Accuracy'])
    #     results[model_name]['Train Accuracy Median'] = np.median(sorted(results[model_name]['Train Accuracy']))
    #     results[model_name]['Train Accuracy Mean'] = np.mean(results[model_name]['Train Accuracy'])
    #     results[model_name]['Train F1 Stdev'] = np.std(results[model_name]['Train F1'])
    #     results[model_name]['Train F1 Median'] = np.median(sorted(results[model_name]['Train F1']))
    #     results[model_name]['Train F1 Mean'] = np.mean(results[model_name]['Train F1'])
    #     results[model_name]['Train Precision Stdev'] = np.std(results[model_name]['Train Precision'])
    #     results[model_name]['Train Precision Median'] = np.median(sorted(results[model_name]['Train Precision']))
    #     results[model_name]['Train Precision Mean'] = np.mean(results[model_name]['Train Precision'])
    #     results[model_name]['Train Recall Stdev'] = np.std(results[model_name]['Train Recall'])
    #     results[model_name]['Train Recall Median'] = np.median(sorted(results[model_name]['Train Recall']))
    #     results[model_name]['Train Recall Mean'] = np.mean(results[model_name]['Train Recall'])

    #     results[model_name]['Val Accuracy Stdev'] = np.std(results[model_name]['Val Accuracy'])
    #     results[model_name]['Val Accuracy Median'] = np.median(sorted(results[model_name]['Val Accuracy']))
    #     results[model_name]['Val Accuracy Mean'] = np.mean(results[model_name]['Val Accuracy'])
    #     results[model_name]['Val F1 Stdev'] = np.std(results[model_name]['Val F1'])
    #     results[model_name]['Val F1 Median'] = np.median(sorted(results[model_name]['Val F1']))
    #     results[model_name]['Val F1 Mean'] = np.mean(results[model_name]['Val F1'])
    #     results[model_name]['Val Precision Stdev'] = np.std(results[model_name]['Val Precision'])
    #     results[model_name]['Val Precision Median'] = np.median(sorted(results[model_name]['Val Precision']))
    #     results[model_name]['Val Precision Mean'] = np.mean(results[model_name]['Val Precision'])
    #     results[model_name]['Val Recall Stdev'] = np.std(results[model_name]['Val Recall'])
    #     results[model_name]['Val Recall Median'] = np.median(sorted(results[model_name]['Val Recall']))
    #     results[model_name]['Val Recall Mean'] = np.mean(results[model_name]['Val Recall'])

    # return results


    
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

# Exec 1
res = {}
file = open('results/feat_sel/selected_features.txt', 'w')
file.close()
file = open('results/feat_sel/models_configs.txt', 'w')
file.close()
res = run_cv('data/data_feat_selection.csv', res, skf, generate_LR = generate_LR, configs_dt = configs_dt, configs_rf = configs_rf, configs_gb = configs_gb, configs_ab = configs_ab)
res = pd.DataFrame.from_dict(res, orient = 'index')
res.to_csv('results/feat_sel/scores.csv')

# Exec 2
# res = pd.read_csv('results/feat_sel/scores.csv', index_col = 0).to_dict(orient = 'index')
# res = run_cv('data/data_one_hot.csv', res, skf, configs_svm = configs_svm, configs_knn = configs_knn, configs_mlp = configs_mlp)
# res = pd.DataFrame.from_dict(res, orient = 'index')
# res.to_csv('results/feat_sel/scores.csv')


print('Time elapsed: {} seconds'.format(time.time() - start_time))
