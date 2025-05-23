import numpy as np
from scipy import sparse


def auto_select_metric(X):
    """
    Автоматический выбор оптимальной метрики расстояния на основе характеристик данных.

    Параметры:
    ----------
    X : array-like
        Входные данные

    Возвращает:
    -------
    str
        Выбранная метрика расстояния
    """
    X = np.asarray(X)

    # Базовые характеристики данных
    n_samples, n_features = X.shape
    is_sparse = sparse.issparse(X)

    # Анализ типов данных
    data_types = [dtype.kind for dtype in X.dtype.descr] if is_sparse else [X.dtype.kind]
    has_categorical = any(kind in 'OS' for kind in data_types)
    has_numeric = any(kind in 'bifc' for kind in data_types)

    # Статистический анализ
    if has_numeric:
        std_dev = np.std(X, axis=0)
        var_coef = np.var(std_dev) / np.mean(std_dev) if np.mean(std_dev) != 0 else 0
        has_outliers = np.any(np.abs(X - np.mean(X, axis=0)) > 3 * std_dev)
    else:
        std_dev = None
        var_coef = 0
        has_outliers = False

    # Выбор метрики на основе характеристик данных
    if has_categorical:
        # Для смешанных данных
        if has_numeric:
            return 'wminkowski'  # Взвешенная метрика Минковского для смешанных типов
        else:
            return 'hamming'  # Для категориальных данных

    # Для числовых данных
    if n_features == 1:
        return 'manhattan'  # Более эффективно для одномерных данных

    if has_outliers:
        return 'chebyshev'  # Лучше обрабатывает выбросы

    if var_coef > 1.0:
        return 'seuclidean'  # Для данных с разной дисперсией

    if is_sparse:
        return 'cityblock'  # Более эффективно для разреженных данных

    # По умолчанию
    return 'euclidean'