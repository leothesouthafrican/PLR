import hdbscan
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

fail_return_dict = {
    'dbcv': -1,
    'rv': -1,
    'silhouette': -1,
    'calinski_harabasz': np.nan,
    'davies_bouldin': np.nan,
    'cluster_count': np.nan,
    'fraction_clustered': 0
}

def cluster_count(data, labels, model=None):
    try:
        return len(np.unique(labels))
    except ValueError:
        print("ValueError caught in cluster_count, returning nan.")
        return fail_return_dict['cluster_count']

def dbcv(data, labels, metric='euclidean', model=None):
    try:
        return hdbscan.validity.validity_index(
            data, labels,
            metric=metric
        )
    except ValueError:
        print("ValueError caught in dbcv, returning -1.")
        return fail_return_dict['dbcv']

def rv(data, labels, metric='euclidean', model=None, pipeline_step=2):
    try:
        return model.steps[pipeline_step][1].relative_validity_
    except AttributeError:
        print("AttributeError caught in rv, returning -1.")
        return fail_return_dict['rv']
    except ValueError:
        print("ValueError caught in rv, returning -1.")
        return fail_return_dict['rv']

def fraction_clustered(data, labels, model=None):
    try:
        fraction = sum(labels == -1) / len(labels)
        return 1 - fraction
    except ValueError:
        print("ValueError caught in fraction_clustered, returning 0.")
        return fail_return_dict['fraction_clustered']

def dbcv_minkowski(data, labels, model=None):
    try:
        return dbcv(data, labels, metric='minkowski')
    except ValueError:
        print("ValueError caught in dbcv_minkowski, returning nan.")
        return fail_return_dict['dbcv']

def silhouette(data, labels, model=None):
    try:
        num_labels = len(set(labels))
        if num_labels == 1:
            print("Warning: Valid number of clusters must be 2 or more.")
            return fail_return_dict['silhouette']
        else:
            return silhouette_score(data, labels)
    except ValueError:
        print("ValueError caught in silhouette, returning -1.")
        return fail_return_dict['silhouette']


def calinski_harabasz(data, labels, model=None):
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


def davies_bouldin(data, labels, model=None):
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
