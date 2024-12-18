from hdbscan.hdbscan_ import *
from sklearn.utils import check_array
import numpy as np


class ClusteringAlgo(BaseEstimator, ClusterMixin):
    def __init__(
            self,
            min_cluster_size=5,
            min_samples=None,
            cluster_selection_epsilon=0.0,
            max_cluster_size=0,
            metric="euclidean",
            alpha=1.0,
            p=None,
            algorithm="best",
            leaf_size=40,
            memory=Memory(None, verbose=0),
            approx_min_span_tree=True,
            gen_min_span_tree=False,
            core_dist_n_jobs=4,
            cluster_selection_method="eom",
            allow_single_cluster=False,
            prediction_data=False,
            match_reference_implementation=False,
            **kwargs
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.alpha = alpha
        self.max_cluster_size = max_cluster_size
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        self.p = p
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.memory = memory
        self.approx_min_span_tree = approx_min_span_tree
        self.gen_min_span_tree = gen_min_span_tree
        self.core_dist_n_jobs = core_dist_n_jobs
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.match_reference_implementation = match_reference_implementation
        self.prediction_data = prediction_data
        self._metric_kwargs = kwargs
        self._condensed_tree = None
        self._single_linkage_tree = None
        self._min_spanning_tree = None
        self._raw_data = None
        self._outlier_scores = None
        self._prediction_data = None
        self._relative_validity = None

    def validate_input_data(self, X):
        X = check_array(X, accept_sparse="csr", force_all_finite=False)
        self._raw_data = X
        return X

    def determine_finiteness(self, X):
        self._all_finite = is_finite(X)

    def extract_finite_data(self, X):
        if not self._all_finite:
            finite_index = get_finite_row_indices(X)
            clean_data = X[finite_index]
            internal_to_raw = {x: y for x, y in zip(range(len(finite_index)), finite_index)}
            outliers = list(set(range(X.shape[0])) - set(finite_index))
        else:
            clean_data = X
            finite_index = None
            internal_to_raw = None
            outliers = None
        return {'finite_index': finite_index, 'clean_data': clean_data, 'internal_to_raw': internal_to_raw,
                'outliers': outliers}

    def prepare_data(self, X):
        X = self.validate_input_data(X)
        self.determine_finiteness(X)
        return self.extract_finite_data(X)

    def perform(self, data, kwargs):
        (
            self.labels_,
            self.probabilities_,
            self.cluster_persistence_,
            self._condensed_tree,
            self._single_linkage_tree,
            self._min_spanning_tree,
        ) = hdbscan(data['clean_data'], **kwargs)

    def unzip_data(self, X):
        kwargs = self.get_params()
        kwargs.pop("prediction_data", None)
        kwargs.update(self._metric_kwargs)
        data = self.prepare_data(X)
        return data, kwargs

    def update_trees(self, data):
        self._condensed_tree = remap_condensed_tree(
            self._condensed_tree, data['internal_to_raw'], data['outliers']
        )
        self._single_linkage_tree = remap_single_linkage_tree(
            self._single_linkage_tree, data['internal_to_raw'], data['outliers']
        )

    def update_labels_and_probabilities(self, data):
        new_labels = np.full(self.X.shape[0], -1)
        new_labels[data['finite_index']] = self.labels_
        self.labels_ = new_labels

        new_probabilities = np.zeros(self.X.shape[0])
        new_probabilities[data['finite_index']] = self.probabilities_
        self.probabilities_ = new_probabilities

    def predict(self, X, data):
        if not self._all_finite:
            self.update_trees(data)
            self.update_labels_and_probabilities(data)
        return self.labels_

    def fit(self, X):
        data, kwargs = self.unzip_data(X)
        self.perform(data, kwargs)
        self.predict(X, data)
        return self

    def fit_another(self, X_train, X_test):
        self.labels_ = []
        data_train, kwargs = self.unzip_data(X_train)
        data_test, _ = self.unzip_data(X_test)
        self.perform(data_train, kwargs)
        self.predict(data_train, data_test)
        return self

    def fit_with_labels(self, X):
        self.fit(X)
        return self.labels_

    def fit_with_labels_another(self, X_train, X_test):
        self.fit_another(X_train, X_test)
        return self.labels_[:len(X_test)]