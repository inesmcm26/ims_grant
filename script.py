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
