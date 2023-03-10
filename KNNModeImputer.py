class KNNModeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5, metric='euclidean', feat_to_impute = None, feats_nn = None):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.feats_nn = feats_nn
        self.feat_to_impute = feat_to_impute

    def fit(self, X, y=None):
        self.X_ = X
        self.is_fitted_ = True
        return self

    def transform(self, X):

      if self.is_fitted_ == False:
        raise Exception("This KNNModeImputer instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

      # create kdtree
      tree = KDTree(self.X_[self.feats_nn], metric=self.metric)

      imputed_X = X.copy()

      missing_rows = X[X[self.feat_to_impute].isna()]

      _, indexes = tree.query(missing_rows[self.feats_nn].values, k= self.n_neighbors) # get indexes of k nearest neighbors

      missing_idx = np.array(missing_rows.index)

      for m_idx, n_idx in enumerate(indexes):
        k_neighbors = self.X_.loc[n_idx, self.feat_to_impute]
        mode = Counter(k_neighbors).most_common(1)[0][0]
        imputed_X.loc[missing_idx[m_idx], self.feat_to_impute] = mode

      return imputed_X