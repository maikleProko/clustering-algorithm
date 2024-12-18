from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.pipeline import Pipeline
import numpy as np


class Estimator:
    def __init__(self, clustering_algo, n_folds=5):
        self.clustering_algo = clustering_algo
        self.n_folds = n_folds

    def evaluate(self, X):
        kf = KFold(n_splits=self.n_folds)
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            labels = self.clustering_algo.fit_with_labels_another(X_train, X_test)

            if len(set(labels)) > 1:
                silhouette_avg = silhouette_score(X_test, labels)
                db_score = davies_bouldin_score(X_test, labels)
                calinski_harabasz = calinski_harabasz_score(X_test, labels)

                silhouette_scores.append(silhouette_avg)
                davies_bouldin_scores.append(db_score)
                calinski_harabasz_scores.append(calinski_harabasz)

        return {
            'silhouette_scores': silhouette_scores,
            'davies_bouldin_scores': davies_bouldin_scores,
            'avg_silhouette_score': np.mean(silhouette_scores) if silhouette_scores else None,
            'avg_davies_bouldin_score': np.mean(davies_bouldin_scores) if davies_bouldin_scores else None,
            'calinski_harabasz_scores': np.mean(calinski_harabasz_scores) if calinski_harabasz_scores else None,
        }

