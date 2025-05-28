# main.py
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from robust_clustering import RobustHierarchicalClusteringModel
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


class Estimator:
    """Класс для оценки качества кластеризации"""

    def __init__(self, X: np.ndarray, labels: np.ndarray):
        self.X = X
        self.labels = labels

    def calculate_metrics(self, n_clusters) -> Dict[str, float]:
        """Рассчёт метрик качества кластеризации"""
        try:
            if len(np.unique(self.labels)) < 2:
                raise ValueError("Необходимо минимум 2 кластера")

            return {
                'silhouette_score': silhouette_score(self.X, self.labels),
                'davies_bouldin_score': davies_bouldin_score(self.X, self.labels),
                'calinski_harabasz_score': calinski_harabasz_score(self.X, self.labels),
                'n_clusters': n_clusters
            }
        except Exception as e:
            print(f"Ошибка при расчёте метрик: {str(e)}")
            return {}


class DataProcessor:
    """Класс для обработки данных"""

    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """Загрузка данных из файла"""
        try:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                return pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                return pd.read_json(file_path)
            raise ValueError(f"Неподдерживаемый формат файла: {file_path}")
        except Exception as e:
            raise ValueError(f"Ошибка при загрузке файла: {str(e)}")

    @staticmethod
    def prepare_features(df: pd.DataFrame, text_columns: Optional[List[str]] = None) -> np.ndarray:
        """Подготовка признаков для кластеризации"""

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for column in df.columns:
            if df[column].dtype == object:
                df[column] = le.fit_transform(df[column])
        return df.values


class ClusteringPipeline:
    """Класс для управления процессом кластеризации"""

    def __init__(self, text_columns: Optional[List[str]] = None):
        self.text_columns = text_columns
        self.model = RobustHierarchicalClusteringModel(
            text_features=text_columns,
            vectorizer_params={
                'max_features': 5000,
                'ngram_range': (1, 2)
            }
        )

    def find_optimal_clusters(self, X: np.ndarray, min_clusters: int = 4, max_clusters: int = 100) -> Dict:
        """Поиск оптимального количества кластеров"""
        best_metrics = None

        for n_clusters in range(min_clusters, max_clusters + 1):
            try:
                self.model.clusterer.n_clusters = n_clusters
                labels = self.model.fit_predict(X)

                estimator = Estimator(X, labels)
                metrics = estimator.calculate_metrics(n_clusters)
                print(metrics)

                if self._is_satisfactory(metrics):
                    best_metrics = {
                        'n_clusters': n_clusters,
                        'metrics': metrics,
                        'status': 'optimal'
                    }
                    break

            except Exception:
                continue

        if not best_metrics:
            raise ValueError("Не найдено удовлетворительных параметров кластеризации")

        return best_metrics

    @staticmethod
    def _is_satisfactory(metrics: Dict[str, float]) -> bool:
        """Проверка качества метрик"""
        silhouette_threshold = -1
        davies_bouldin_threshold = 1.0
        calinski_harabasz_threshold = 10
        check = (
                metrics['davies_bouldin_score'] <= davies_bouldin_threshold and
                metrics['calinski_harabasz_score'] >= calinski_harabasz_threshold or

                metrics['silhouette_score'] >= silhouette_threshold and
                metrics['calinski_harabasz_score'] >= calinski_harabasz_threshold or

                metrics['silhouette_score'] >= silhouette_threshold and
                metrics['davies_bouldin_score'] <= davies_bouldin_threshold
        )

        return check


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Кластеризация данных')
    parser.add_argument('--input_file', required=True, help='Путь к входному файлу с данными')
    parser.add_argument('--n_clusters', type=int, default=8, help='Количество кластеров')
    parser.add_argument('--text_columns', nargs='+', help='Имена текстовых столбцов')
    parser.add_argument('--output_file', default='clusters.csv', help='Файл для сохранения результатов')

    args = parser.parse_args()

    try:
        # Загрузка и подготовка данных
        df = DataProcessor.load_data(args.input_file)
        X = DataProcessor.prepare_features(df, args.text_columns)

        # Кластеризация
        pipeline = ClusteringPipeline(args.text_columns)
        best_params = pipeline.find_optimal_clusters(X)

        # Финальная кластеризация
        pipeline.model.clusterer.n_clusters = best_params['n_clusters']
        labels = pipeline.model.fit_predict(X)

        # Сохранение результатов
        result_df = df.copy()
        result_df['cluster'] = labels
        print(result_df)
        result_df.to_csv(args.output_file, index=False)

        # Вывод метрик
        print("\nФинальные метрики качества кластеризации:")
        for name, value in best_params['metrics'].items():
            print(f"{name}: {value:.3f}")


        # Подсчет элементов в кластерах
        print("\nРаспределение элементов по кластерам:")
        cluster_counts = result_df['cluster'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            print(f"Кластер {cluster_id}: {count} элементов")

        pipeline.model.visualize_dendrogram()
        pipeline.model.visualize_scatter()

    except Exception as e:
        print(f"Ошибка при обработке данных: {str(e)}")


if __name__ == "__main__":
    main()