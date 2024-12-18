from sklearn.preprocessing import LabelEncoder
import clustering_algo
from clustering_algo import ClusteringAlgo
from sklearn.datasets import make_blobs
import pandas as pd

from estimator import Estimator


def convert_categorical_to_numeric(df):
    le = LabelEncoder()

    for column in df.columns:
        if df[column].dtype == object:
            df[column] = le.fit_transform(df[column])
    return df


if __name__ == '__main__':
    data = pd.read_csv('survey lung cancer.csv')
    print('LOADING DATA')
    print(data)
    converted_data = convert_categorical_to_numeric(data)
    print('\nFIRST PREPARING DATA')
    print(converted_data)
    clusterer = clustering_algo.ClusteringAlgo()
    cluster_labels = clustering_algo.ClusteringAlgo().fit_predict(data)
    print('\nCLUSTERING RESULT')
    print(cluster_labels)
    print('\nEVALUATE')
    estimator = Estimator(clusterer)
    clustering_algorithm = ClusteringAlgo(min_cluster_size=5)
    estimator = Estimator(clustering_algo=clustering_algorithm)
    results = estimator.evaluate(converted_data.to_numpy())
    print("Silhouette Scores:", results['silhouette_scores'])
    print("Davies-Bouldin Scores:", results['davies_bouldin_scores'])
    print("Average Silhouette Score:", results['avg_silhouette_score'])
    print("Average Davies-Bouldin Score:", results['avg_davies_bouldin_score'])
    print("Calinski Harabasz Score:", results['calinski_harabasz_scores'])
