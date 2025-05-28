# robust_clustering.py
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
from typing import Optional, Union, Tuple, List
import joblib
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


class RobustHierarchicalClusteringModel:
    def __init__(
            self,
            imputation_k: int = 5,
            scaling_with_median: bool = True,
            linkage: str = 'ward',
            distance_threshold: Optional[float] = None,
            text_features: Optional[List[str]] = None,
            vectorizer_params: Optional[dict] = None
    ):
        """
        Инициализация модели иерархической кластеризации с поддержкой неструктурированных данных.

        Args:
            imputation_k: Количество соседей для импутации
            scaling_with_median: Использовать робастное масштабирование
            linkage: Метод связности ('ward', 'complete', 'average', 'single')
            distance_threshold: Пороговое расстояние для остановки слияния кластеров
            text_features: Список имен текстовых признаков
            vectorizer_params: Параметры векторизатора текста
        """
        self.imputer = KNNImputer(n_neighbors=imputation_k)
        self.scaler = RobustScaler(with_centering=not scaling_with_median)
        self.clusterer = AgglomerativeClustering(
            linkage=linkage,
            distance_threshold=distance_threshold,
            compute_distances=True
        )
        vectorizer_params = vectorizer_params or {
            'max_features': 10000,
            'ngram_range': (1, 2),
            'sublinear_tf': True
        }
        self.vectorizer = TfidfVectorizer(**vectorizer_params)
        self.text_features = text_features
        self.is_fitted = False
        self.partial_fit_data = None
        self.linkage_matrix = None

    def _prepare_features(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка признаков для кластеризации."""
        features = None
        if self.text_features:
            numerical_features = X[:, [i for i in range(X.shape[1])
                                       if i not in self.text_features]]
            text_features = X[:, self.text_features]
            text_vectors = self.vectorizer.fit_transform(text_features)
            text_features = text_vectors.toarray()
            features = np.hstack([numerical_features, text_features])
        else:
            features = X
        return features

    def fit(self, X: np.ndarray) -> 'RobustHierarchicalClusteringModel':
        """Обучение модели на данных."""
        features = self._prepare_features(X)
        X_imputed = self.imputer.fit_transform(features)
        X_scaled = self.scaler.fit_transform(X_imputed)
        self.clusterer.fit(X_scaled)
        self.linkage_matrix = linkage(X_scaled, method=self.clusterer.linkage)
        self.is_fitted = True
        self.partial_fit_data = (X_imputed, X_scaled)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание кластеров для новых данных."""
        if not self.is_fitted:
            raise ValueError("Модель не обучена")
        features = self._prepare_features(X)
        X_imputed = self.imputer.transform(features)
        X_scaled = self.scaler.transform(X_imputed)
        return self.clusterer.fit_predict(X_scaled)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def partial_fit(self, X: np.ndarray) -> 'RobustHierarchicalClusteringModel':
        """Частичное обучение модели на новых данных."""
        if not self.is_fitted:
            return self.fit(X)

        features = self._prepare_features(X)
        X_imputed = self.imputer.transform(features)
        X_scaled = self.scaler.transform(X_imputed)

        # Объединяем новые данные с существующими
        if self.partial_fit_data:
            X_imputed = np.vstack([self.partial_fit_data[0], X_imputed])
            X_scaled = np.vstack([self.partial_fit_data[1], X_scaled])

        self.clusterer.fit(X_scaled)
        self.linkage_matrix = linkage(X_scaled, method=self.clusterer.linkage)
        self.partial_fit_data = (X_imputed, X_scaled)
        return self

    def visualize_dendrogram(self, figsize: Tuple[int, int] = (10, 7)):
        """Визуализация дендрограммы."""
        if not self.is_fitted:
            raise ValueError("Модель не обучена")

        plt.figure(figsize=figsize)
        dendrogram(self.linkage_matrix)
        plt.title('Дендрограмма иерархической кластеризации')
        plt.xlabel('Индекс образца')
        plt.ylabel('Расстояние')
        plt.show()

    def visualize_scatter(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Визуализация графика рассеяния кластеров.

        Args:
            figsize: Размер графика (ширина, высота)
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена")

        if self.partial_fit_data:
            X_scaled = self.partial_fit_data[1]
        else:
            X_scaled = self.clusterer.cluster_centers_

        # Если больше 2 признаков, используем первые два
        if X_scaled.shape[1] > 2:
            X_scaled = X_scaled[:, :2]

        plt.figure(figsize=figsize)
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c='blue', alpha=0.5)
        plt.title('График рассеяния кластеров')
        plt.xlabel('Первый признак')
        plt.ylabel('Второй признак')
        plt.grid(True)
        plt.show()

    def save(self, path: str) -> None:
        """Сохранение модели в файл."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> 'RobustHierarchicalClusteringModel':
        """Загрузка модели из файла."""
        return joblib.load(path)