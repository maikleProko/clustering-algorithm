from sklearn.preprocessing import LabelEncoder
import clustering_algo
from clustering_algo import ClusteringAlgo
from sklearn.datasets import make_blobs
import pandas as pd

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
    print(len(cluster_labels))


