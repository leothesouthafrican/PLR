import hdbscan
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

fail_return_dict = {
    'dbcv': -1,
    'silhouette': -1,
    'calinski_harabasz': np.nan,
    'davies_bouldin': np.nan,
    'cluster_count': np.nan
}

def cluster_count(data, labels):
    try:
        return len(np.unique(labels))
    except ValueError:
        print("ValueError caught in cluster_count, returning nan.")
        return fail_return_dict['cluster_count']

def dbcv(data, labels, metric='euclidean'):
    try:
        return hdbscan.validity.validity_index(
            data, labels,
            metric=metric
        )
    except ValueError:
        print("ValueError caught in dbcv, returning nan.")
        return fail_return_dict['dbcv']


def dbcv_minkowski(data, labels):
    try:
        return dbcv(data, labels, metric='minkowski')
    except ValueError:
        print("ValueError caught in dbcv_minkowski, returning nan.")
        return fail_return_dict['dbcv']

def silhouette(data, labels):
    try:
        num_labels = len(set(labels))
        if num_labels == 1:
            print("Warning: Valid number of clusters must be 2 or more.")
            return fail_return_dict['silhouette']
        else:
            return silhouette_score(data, labels)
    except ValueError:
        print("ValueError caught in silhouette, returning nan.")
        return fail_return_dict['silhouette']


def calinski_harabasz(data, labels):
    try:
        num_labels = len(set(labels))
        if num_labels == 1:
            print("Warning: Valid number of clusters must be 2 or more.")
            return fail_return_dict['calinski_harabasz']
        else:
            return calinski_harabasz_score(data, labels)
    except ValueError:
        print("ValueError caught in calinski_harabasz, returning nan.")
        return fail_return_dict['calinski_harabasz']


def davies_bouldin(data, labels):
    """
    Note: 0 is best. If using for CV need to use complement.
    """
    try:
        num_labels = len(set(labels))
        if num_labels == 1:
            print("Warning: Valid number of clusters must be 2 or more.")
            return fail_return_dict['davies_bouldin']
        else:
            return davies_bouldin_score(data, labels)
    except ValueError:
        print("ValueError caught in davies_bouldin, returning nan.")
        return fail_return_dict['davies_bouldin']
