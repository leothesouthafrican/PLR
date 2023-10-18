import hdbscan
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def dbcv(data, labels, metric='euclidean'):
    return hdbscan.validity.validity_index(
        data, labels,
        metric=metric
    )


def dbcv_minkowski(data, labels):
    return dbcv(data, labels, metric='minkowski')


def silhouette(data, labels):
    num_labels = len(set(labels))
    if num_labels == 1:
        print("Warning: Valid number of clusters must be 2 or more.")
        return np.nan
    else:
        return silhouette_score(data, labels)


def calinski_harabasz(data, labels):
    num_labels = len(set(labels))
    if num_labels == 1:
        print("Warning: Valid number of clusters must be 2 or more.")
        return np.nan
    else:
        return calinski_harabasz_score(data, labels)


def davies_bouldin(data, labels):
    """
    Note: 0 is best. If using for CV need to use complement.
    """
    num_labels = len(set(labels))
    if num_labels == 1:
        print("Warning: Valid number of clusters must be 2 or more.")
        return np.nan
    else:
        return davies_bouldin_score(data, labels)
