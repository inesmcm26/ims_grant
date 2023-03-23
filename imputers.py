import numpy as np
from scipy.stats import mode
from sklearn.impute import KNNImputer

# Custom KNNImputer that returns de mode of the k nearest neighbors and can deal with missing values when calculating the distance
class KNNImputerMode(KNNImputer):
    def __init__(self, missing_values = np.nan, n_neighbors = 5, weights = "uniform", metric = "nan_euclidean", copy = True, add_indicator = False):
        super().__init__(missing_values = missing_values, n_neighbors = n_neighbors, weights = weights,
                         metric = metric, copy = copy, add_indicator = add_indicator)

    def _calc_impute(self, dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col):
        # Get donors
        donors_idx = np.argpartition(dist_pot_donors, n_neighbors - 1, axis=1)[
            :, :n_neighbors
        ]

        # Retrieve donor values and calculate kNN mode
        donors = fit_X_col.take(donors_idx)
        donors_mask = mask_fit_X_col.take(donors_idx)
        donors = np.ma.array(donors, mask=donors_mask)
        
        return np.ravel(mode(donors, axis=1).mode)


# Custom KNNImputer that returns de median of the k nearest neighbors and can deal with missing values when calculating the distance
class KNNImputerMedian(KNNImputer):
    def __init__(self, missing_values = np.nan, n_neighbors = 5, weights = "uniform", metric = "nan_euclidean", copy = True, add_indicator = False):
        super().__init__(missing_values = missing_values, n_neighbors = n_neighbors, weights = weights,
                         metric = metric, copy = copy, add_indicator = add_indicator)

    def _calc_impute(self, dist_pot_donors, n_neighbors, fit_X_col, mask_fit_X_col):
        # Get donors
        donors_idx = np.argpartition(dist_pot_donors, n_neighbors - 1, axis=1)[
            :, :n_neighbors
        ]

        # Retrieve donor values and calculate kNN mode
        donors = fit_X_col.take(donors_idx)
        donors_mask = mask_fit_X_col.take(donors_idx)
        donors = np.ma.array(donors, mask=donors_mask)
        
        return np.median(donors, axis=1)