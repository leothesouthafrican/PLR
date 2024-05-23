import json
import hdbscan
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score

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

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def replace_noise(labels, ignore_label):

    max_l = np.max(labels)
    _labels = []

    for l in labels:
        if l == ignore_label:
            _labels.append(max_l + 1)
            max_l += 1
        else:
            _labels.append(l)

    return _labels


def clustering_similarity(labels_1, labels_2, ignore_label=-1, score='rand', _replace_noise=True):
    assert len(labels_1) == len(labels_2)

    if ignore_label is not None:
        if _replace_noise:
            _labels_1 = replace_noise(labels_1, ignore_label)
            _labels_2 = replace_noise(labels_2, ignore_label)
        else:
            keep_indices = list(set(
                np.argwhere(labels_1 != ignore_label).flatten()
            ).union(np.argwhere(labels_2 != ignore_label).flatten()))

            _labels_1 = labels_1[keep_indices]
            _labels_2 = labels_2[keep_indices]

    else:
        _labels_1 = labels_1
        _labels_2 = labels_2

    if score == 'rand':
        score = adjusted_rand_score
    elif score == 'mi':
        score = adjusted_mutual_info_score
    elif score == 'norm':
        score = normalized_mutual_info_score

    return score(_labels_1, _labels_2)