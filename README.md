# Roadmap

## Repository organization:

- `data`: Where different versions of the dataset are saved after preprocessing steps
- `Finacial Well-Being Survey data`: Contains the original dataset and pdfs with metadata
- `relatorios`: FCT reports
- `results`: Contains two subfolders, one for each grid search run. Each sobfolder contains a text file with all the model configurations that were tested + the resulting scores for each configuration across the cross-validation folds. Besided from that there is:
    - `feat_sel/selected_features.csv`: which features were choosen in each CV split.
    - `original/tpot_pipeline.py`: the file that contains the optimal pipeline found by TPOT.
- `data_exploration.ipynb`: Exploration of the dataset
- `experimental_results.ipynb`: Analysis of the AutoML experimental results. In this notebook, the scores obtained by each Grid Search run and TPOT are statistically compared.
- `feat_engineering.ipynb`: Feature engineered dataset creation
- `feat_selection.ipynb`: Feature selection dataset creation
- `imputers.py`: Data imputation classes
- `models.py`: Auxiliary file with functions to create random model configurations and respective instances to be used on the Grid Search.
- `preprocess.ipynb`: All preprocessing steps on the original dataset. These include data cleaning, missing values treatment, categorical variables one-hot encoding and numerical variables scaling.
- `scalers.py`: File with scaling methods: MinMaxScaler and StarndardScaler for features with different distributions.
- `script_feat_sel.py`: Grid Search script for dataset that has gone through feature engineering. In this grid search, feature selection is performed in each CV fold using RFE.
- `script.py`: Grid Search script for original dataset. No feature selection or feature engineering are performed in this script.

## Project goal

This project aims to compare different AutoML approaches to find the best model and respective configuration to solve a classification task.

The original dataset used is the Financial Well-Being Survey dataset conduted by the Consumer Financial Protection Bureau, in the United States in 2017. The target is a categorical variable with 3 possible values.

Given the target imbalance, the metrics used to evaluate the model's performance are the F1-Score, Precision, Recall and Accuracy.

Two Grid Search were run:
- `script.py`: The first one with the original dataset after being cleaned. This dataset has only gone through the steps in `preprocessing.ipynb`.
- `script_feat_sel.py`: The second one with the cleaned dataset and feature engineering. This dataset has gone through the steps on `feat_engineering.ipynb` where new features were created, and the `feat_selection.ipynb` where some global irrelevant features were removed à priori given a correlation-based criteria. Besides from that, on each cross-validation fold, RFE was used to select the best features and all the other ones were discarded. The features selected in each CV fold were saved on the `selected_features.csv` file on the `results/feat_sel` folder.

In both scripts, the model configurations to be tested are firstly defined and then the models instances are created. These configurations are randomly generated from within some pre-defined hyperparameters space. After that, these models are trained and tested in each CV fold and the results are saved.

To change the models to be tested and the respective hyperparameter space, the `models.py` file can be edited on the `generate_configs_<model>` functions. After that, the `script.py` and `script_feat_sel.py` files also need to be updated to generate the desired model configurations. The number of different configurations of the same model to be generated can be changed on the parameter `n_models` on the `generate_configs_<model>` function calls on the scripts file. After that, these configurations are passed to the `get_models` function in each fold, in order to generate new instances of the same model configuration in each CV fold.

Apart from the Grid Searches, TPOT was also tested. The dataset used was the same used for the first Grid Search. Since TPOT does its own feature selection and feature engineering, the cleaned dataset was the ideal one to give to TPOT. Also, given that TPOT does its own cross-validation during the optimization, it was run only once. After that, the returned optimal pipeline was tested using the same CV folds as all the other tested models of the Grid Search. This is the only way to fairly compare the optimal pipeline returned by TPOT with the models tested using the Grid Search Cross-Validation. The code for TPOT is also on the `script.py` file.


## Report

For more contextualized and detailed information about the preprocessing steps, the Grid Search and TPOT you can consult the `Report_AICE_InesMagessi.pdf` file on the `relatorios` folder.