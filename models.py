import random
import itertools

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier # XGBoost
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from tpot import TPOTClassifier

import numpy as np

# Random Seed
np.random.seed(0)
random.seed(0)
seed = 0

def get_random(n_models, *args):
    # Generate all possible combinations of parameter values
    all_configs = list(itertools.product(*args))

    # Randomly sample n_models number of configurations
    random_configs = random.sample(all_configs, n_models)

    return random_configs


def generate_configs_DT(n_models):

    criterion = ['gini', 'entropy']
    max_depths = [5, 10, 50, 100, None]
    min_samples_splits = [2, 5, 15]
    max_features = [0.5, 0.9, 'sqrt', 'log2', None]

    return get_random(n_models, criterion, max_depths, min_samples_splits, max_features)

def generate_configs_RF(n_models):

    n_estimators = [50, 100, 200, 350]
    criterion = ['gini', 'entropy']
    max_depths = [10, 50, 100, None]
    min_samples_splits = [2, 5, 15]
    max_features = [0.5, 0.7, 0.9, 'sqrt', 'log2', None]
    bootstrap = [True, False]
    max_samples = [0.5, 0.7, 0.9, None]

    # Generate all possible combinations of parameter values
    all_configs = list(itertools.product(n_estimators, criterion, max_depths, min_samples_splits, max_features, bootstrap, max_samples))

    random.shuffle(all_configs)

    configs = []

    for config in all_configs:

        if [config[0], config[1], config[2], config[3], config[4], config[5]] in configs:
            continue
        else:
            configs.append([config[0], config[1], config[2], config[3], config[4], config[5], config[6]])
            if len(configs) == n_models:
                break

    return configs

def generate_configs_GB(n_models):
    learning_rates = [0.1, 0.01, 0.001]
    subsamples = [0.5, 0.7, 0.9, 1]
    n_estimators = [50, 100, 200, 350]
    min_samples_splits = [2, 5, 15]
    max_depths = [1, 3, 10]
    max_features = [0.5, 0.9, 'sqrt', 'log2', None]

    return get_random(n_models, learning_rates, subsamples, n_estimators, min_samples_splits, max_depths, max_features)

def generate_configs_AB(n_models):
    n_estimators = [20, 50, 100, 200, 350]
    learning_rates = [1, 5, 10]

    return get_random(n_models, n_estimators, learning_rates)

def generate_configs_SVC(n_models):
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    Cs = [0.1, 1.0, 10.0, 100.0]
    degrees = [2, 3, 4]
    gammas = [0.1, 1, 10, 100]

    return get_random(n_models, kernels, Cs, degrees, gammas)

def generate_configs_KNN(n_models):
    n_neighbors = [3, 5, 7, 9]
    weights = ['uniform', 'distance']
    metric = ['euclidean', 'manhattan']

    return get_random(n_models, n_neighbors, weights, metric)

def generate_configs_MLP(n_models):
    hidden_layer_sizes = [(10,), (10, 10, 10), (32, 64, 128), (100, 100), (200, 350, 100)]
    alpha = [0.0001, 0.001, 0.01]
    batch_size = [1, 32, 64, 128, 256]
    learning_rate_init = [0.0001, 0.001]   

    return get_random(n_models, hidden_layer_sizes, alpha, batch_size, learning_rate_init)

def generate_DT(configs, class_weights, seed):
    models = {}

    for i, config in enumerate(configs):
        models['Decision Tree ' + str(i+1)] = DecisionTreeClassifier(
            criterion=config[0],
            max_depth=config[1],
            min_samples_split=config[2],
            max_features=config[3],
            class_weight = class_weights,
            random_state = seed
        )

    return models

def generate_RF(configs, class_weights, seed):

    models = {}

    for i, config in enumerate(configs):
        models['Random Forest ' + str(i+1)] = RandomForestClassifier(
            n_estimators=config[0],
            criterion=config[1],
            max_depth=config[2],
            min_samples_split=config[3],
            max_features=config[4],
            bootstrap=config[5],
            max_samples=config[6] if config[5] == True else None,
            random_state = seed,
            class_weight = class_weights
        )

    return models

def generate_GB(configs, seed):

    models = {}

    for i, config in enumerate(configs):
        models['Gradient Boosting ' + str(i+1)] = GradientBoostingClassifier(
            learning_rate=config[0],
            subsample=config[1],
            n_estimators=config[2],
            min_samples_split=config[3],
            max_depth=config[4],
            max_features=config[5],
            random_state = seed,
        )

    return models

def generate_AB(configs, seed):

    models = {}

    for i, config in enumerate(configs):
        models['AdaBoost ' + str(i+1)] = AdaBoostClassifier(
            n_estimators = config[0],
            learning_rate = config[1],
            random_state = seed
        )

    return models

def generate_SVC(configs, class_weights, seed):

    models = {}

    for i, config in enumerate(configs):
        models['SVC ' + str(i+1)] = SVC(
            kernel = config[0],
            C = config[1],
            degree = config[2],
            gamma = config[3],
            decision_function_shape = 'ovo',
            random_state = seed,
            class_weight = class_weights
        )

    return models

def generate_KNN(configs):

    models = {}

    for i, config in enumerate(configs):
        models['KNN ' + str(i+1)] = KNeighborsClassifier(
            n_neighbors = config[0],
            weights = config[1],
            metric = config[2]
        )

    return models

def generate_MLP(configs, seed):

    models = {}

    for i, config in enumerate(configs):
        models['MLP ' + str(i+1)] = MLPClassifier(
            hidden_layer_sizes = config[0],
            alpha = config[1],
            batch_size = config[2],
            learning_rate_init = config[3],
            random_state = seed,
            verbose = True
        )

    return models

def generate_LR(seed, class_weights):
    models = {
    # Logistic Regression: 2 models {class_label: weight}
    'Logistic Regression 1': LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs', class_weight = class_weights, random_state = seed),
    'Logistic Regression 2': LogisticRegression(multi_class = 'multinomial', solver = 'sag', class_weight = class_weights, random_state = seed),
    }

    return models


def get_models(generate_LR = False, configs_dt = None, configs_rf = None, configs_gb = None, configs_ab = None, configs_svm = None, configs_knn = None,
               configs_mlp = None, class_weights = None, seed = None):

    models = {}

    if generate_LR == True:
        models.update(generate_LR(seed), class_weights)

    if configs_dt != None:
        # Decision Tree: 150 possible models class_weight = {class_label: weight}
        dt = generate_DT(configs_dt, class_weights, seed)
        models.update(dt)

    if configs_rf != None:
        # Random Forest: 2880 possible models class_weight = {class_label: weight}
        rf = generate_RF(configs_rf, class_weights,seed)
        models.update(rf)

    if configs_gb != None:
        # Gradient Boosting: 2160 possible combinations 30/50
        gb = generate_GB(configs_gb, seed = seed)
        models.update(gb)

    if configs_ab != None:
        # Adaptive Boosting: 15 possible combinations
        ab = generate_AB(configs_ab, seed = seed)
        models.update(ab)
    
    if configs_svm != None:
        # SVM: 96 possible combinations
        svc = generate_SVC(configs_svm, class_weights = class_weights, seed = seed)
        models.update(svc)

    if configs_knn != None:
        # KNN: 16 possible combinations
        knn = generate_KNN(configs_knn)
        models.update(knn)
    
    if configs_mlp != None:
        # Neural Network: 150 possible models NO SAMPLE_WEIGHT
        mlp = generate_MLP(configs_mlp, seed = seed)
        models.update(mlp)

    return models