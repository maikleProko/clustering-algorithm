from sklearn.preprocessing import LabelEncoder
import clustering_algo
from clustering_algo import ClusteringAlgo
from sklearn.datasets import make_blobs
import pandas as pd
import argparse
import time
import psutil
import os
from auto_select_metric import auto_select_metric

from estimator import Estimator


def convert_categorical_to_numeric(df):
    le = LabelEncoder()

    for column in df.columns:
        if df[column].dtype == object:
            df[column] = le.fit_transform(df[column])
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run clustering on a CSV file.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file.')
    parser.add_argument('clustering_method', type=str, nargs='?', default='',
                       help='Name of the clustering method to use (e.g., "AgglomerativeClustering").')
    parser.add_argument('--mode', type=str, choices=['memory', 'time'],
                       help='Additional information to display: memory usage or execution time')
    args = parser.parse_args()
    no_error_with_file = True

    data = None

    try:
        data = pd.read_csv(args.csv_file)

        print('LOADING DATA')
        print(data)
        converted_data = convert_categorical_to_numeric(data)
    except:
        no_error_with_file = False
        print('Error with work with file: ' + str(args.csv_file))

    metric = args.clustering_method
    if not metric:
        metric = auto_select_metric(data.to_numpy())

    if no_error_with_file:
        clusterer = None
        cluster_labels = None
        try:
            clusterer = clustering_algo.ClusteringAlgo(metric=metric)
            cluster_labels = clusterer.fit_predict(data)
        except:
            print('Error with work with clustering method: ' + str(args.clustering_method))

        print('\nCLUSTERING RESULT')
        print(cluster_labels)

        # Добавлен код для отслеживания памяти и времени
        if args.mode == 'memory':
            process = psutil.Process(os.getpid())
            mem_usage = process.memory_info().rss / (1024 * 1024)  # в МБ
            print(f'\nMemory usage: {mem_usage:.2f} MB')
        elif args.mode == 'time':
            print(f'\nExecution time: {time.process_time():.2f} seconds')