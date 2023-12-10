"""
Implements the ensemble selection methods JC and CAS from Fern 2008.

First produces a set of base results. Then re-runs the method N_REPEATS times,
with different random seeds, comparing the output to the base results.
# TODO: implement CAS method.
# TODO: implement and run with hdbscan only - for noise clusters.
"""
import pickle
import pickle as pk
import sys
from itertools import combinations

import pandas as pd
import numpy as np
import os
from pathlib import Path
import networkx as nx
from scipy.cluster import hierarchy
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score

from utilities import run_configs, load_symptom_data, modularity, clustering_similarity

ENSEMBLE_SELECTION_ID = int(sys.argv[1])
CLUSTERING_ALGO = str(sys.argv[2])

ENSEMBLE_SIZE = 20
LIBRARY_N_CLUSTER = 3
N_REPEATS = 10
SEARCH_TYPE = 'randomized_search'  # we want random parameterisations for diversity.

if CLUSTERING_ALGO == 'kmeans':
    SAMPLE_SIZE = 200  # number of sample to take from each pipeline to build library
    RUN_IDS_TO_INCLUDE = [1, 2, 5, 6]  # we will reproduce using only kmeans (and including p-umap)
elif CLUSTERING_ALGO == 'hdbscan':
    SAMPLE_SIZE = 15  # number of sample to take from each pipeline to build library
    RUN_IDS_TO_INCLUDE = [3, 4, 7, 8]  # we will reproduce using only kmeans (and including p-umap)

NMI_SCORE = 'mi'  # arg to pass to clustering_similarity method to use partial NMI (ignoring -1 labels from hdbscan)
BASE_SEED = 0  # random seed for base results
ENSEMBLE_SELECTION_METHODS = ['JC', 'CAS']
ALPHA = 0.5  # JC objective weighting
# Note: set IGNORE_LABEL to None for speed, unless in use:
IGNORE_LABEL = None  # ignore noise in hdbscan when building co-association matrix and computing similarity scores.
REPLACE_NOISE = False
LINKAGE_METHOD = 'average'
MAXCLUST = 15

ENSEMBLE_SELECTION_METHOD = ENSEMBLE_SELECTION_METHODS[ENSEMBLE_SELECTION_ID]

save_dir = Path(
    './fern_ensemble_outputs/%s_%s_%d_%.1f'
    % (CLUSTERING_ALGO, ENSEMBLE_SELECTION_METHOD, ENSEMBLE_SIZE, ALPHA)
)

os.makedirs(
    save_dir,
    exist_ok=False
)

run_metadata = {
    'bayes_search': {
        1: {
            'run_path': 'rusty-chris/tune_shallow_clustering/runs/dbatkzah',
            'results_path': 'results/bayes_search/umap_kmeans_silhouette_run_10/all_results.pickle'
        },
        3: {
            'run_path': 'rusty-chris/tune_shallow_clustering/runs/ddgyhpw8',
            'results_path': 'results/bayes_search/umap_hdbscan_silhouette_run_10/all_results.pickle'
        },
        5: {
            'run_path': 'rusty-chris/tune_shallow_clustering/runs/eg0fld9h',
            'results_path': 'results/bayes_search/pca_kmeans_silhouette_run_10/all_results.pickle'
        },
        7: {
            'run_path': 'rusty-chris/tune_shallow_clustering/runs/umfnoehu',
            'results_path': 'results/bayes_search/pca_hdbscan_silhouette_run_10/all_results.pickle'
        }
    },
    'randomized_search': {
        1: {
            'run_path': 'rusty-chris/tune_shallow_clustering/runs/ndz8b3cz',
            'results_path': 'results/umap_kmeans_silhouette_run_10/all_results.pickle'
        },
        2: {
            'run_path': 'rusty-chris/tune_shallow_clustering/runs/gcnqmv1k',
            'results_path': 'results/umap_kmeans_dbcv_run_11/all_results.pickle'
        },
        3: {
            'run_path': 'rusty-chris/tune_shallow_clustering/runs/t0tzp4gz',
            'results_path': 'results/umap_hdbscan_silhouette_run_10/all_results.pickle'
        },
        4: {
            'run_path': 'rusty-chris/tune_shallow_clustering/runs/zgsa9yi4',
            'results_path': 'results/umap_hdbscan_dbcv_run_11/all_results.pickle'
        },
        5: {
            'run_path': 'rusty-chris/tune_shallow_clustering/runs/x42hq5ez',
            'results_path': 'results/pca_kmeans_silhouette_run_10/all_results.pickle'
        },
        6: {
            'run_path': 'rusty-chris/tune_shallow_clustering/runs/qgy8ifuf',
            'results_path': 'results/pca_kmeans_dbcv_run_10/all_results.pickle'
        },
        7: {
            'run_path': 'rusty-chris/tune_shallow_clustering/runs/en968oqc',
            'results_path': 'results/pca_hdbscan_silhouette_run_10/all_results.pickle'
        },
        8: {
            'run_path': 'rusty-chris/tune_shallow_clustering/runs/ocncbqca',
            'results_path': 'results/pca_hdbscan_dbcv_run_10/all_results.pickle'
        },
        13: {
            'run_path': 'rusty-chris/tune_shallow_clustering/runs/lo5r07or',
            'results_path': 'results/parametric_umap_kmeans_silhouette_run_10/all_results.pickle'
        },
        15: {
            'run_path': 'rusty-chris/tune_shallow_clustering/tv9hnbnn',
            'results_path': 'results/parametric_umap_hdbscan_silhouette_run_10/all_results.pickle'
        }
    }
}

for key in run_metadata[SEARCH_TYPE].keys():
    run_configs[key].update(run_metadata[SEARCH_TYPE][key])


def load_results(run_id):
    run_config = run_configs[run_id]
    with open(run_config['results_path'], 'rb') as outfile:
        results = pk.load(outfile)

    results = pd.DataFrame.from_dict(results, orient='index')
    results['original_index'] = results.index
    results['run_id'] = run_id
    return results


def filter_results(_results, min_size=2, noise_threshold=0.66):
    _results = _results[_results.cluster_count > min_size]
    _results = _results[_results.fraction_clustered > noise_threshold]

    return _results


def sample_results(_results):
    if len(_results) < SAMPLE_SIZE:
        return _results
    else:
        return _results.sample(SAMPLE_SIZE)


symptom_data = load_symptom_data(run_configs[1]['data_path'])
all_data = pd.read_csv(run_configs[1]['data_path'], index_col=0)
tessa = pd.read_csv('../clusterings/tessa/cluster_13_111023.csv')
tessa.rename(columns={'Unnamed: 0': 'index'}, inplace=True)


def convert_age(age_string):
    conversion_diict = {
        '30-39': 35,
        '40-49': 45,
        '50-59': 55,
        '18-29': 24,
        '60-69': 65,
        '70-79': 75,
        '80+': 85
    }
    return conversion_diict[age_string]


def build_cluster_summary(_all_data, labels):

    symptoms = [
        col
        for col in _all_data.columns
        if 'Symptom' in col
    ]

    _all_data['cluster'] = labels
    _all_data['dummy'] = 1
    _all_data['woman'] = _all_data['Demographics_Gender_Cleaned'] == 'Woman'
    _all_data['Flag_MCAS_norm'] = _all_data['Flag_MCAS'] / 6
    _all_data['symptom_count'] = _all_data[symptoms].sum(axis=1)
    _all_data['age_numeric'] = _all_data['Demographics_Age_Cleaned'].apply(convert_age)
    _all_data['Physical_PEM_Severity_norm'] = _all_data['Physical_PEM_Severity'] / 10
    _all_data['Cognitive_PEM_Severity_norm'] = _all_data['Cognitive_PEM_Severity'] / 10

    cluster_summary = _all_data.groupby('cluster').agg({
        'dummy': len,
        'Flag_MECFS': sum,
        'Flag_POTS': pd.Series.mode,
        'Flag_MCAS_norm': sum,
        'woman': sum,
        'symptom_count': np.median,
        'age_numeric': np.mean,
        'Physical_PEM_Severity_norm': np.mean,
        'Cognitive_PEM_Severity_norm': np.mean
    })
    cluster_summary['Flag_MECFS'] /= cluster_summary['dummy']
    cluster_summary['Flag_MCAS_norm'] /= cluster_summary['dummy']
    cluster_summary['woman'] /= cluster_summary['dummy']
    cluster_summary.rename(columns={'dummy': 'cluster_size'})

    return cluster_summary


def ensemble_to_co_association(ensemble):
    if isinstance(ensemble, pd.DataFrame) and 'labels' in ensemble.columns:
        df = pd.DataFrame(ensemble.labels)
        df = pd.DataFrame(df['labels'].to_list()).transpose()
    else:
        df = pd.DataFrame(ensemble).transpose()

    N = len(df)

    co_association_matrix = np.zeros([N, N])
    n_estimators = len(library)

    for ri, row in df.iterrows():

        if ri % 500 == 0:
            print(ri)

        compare = df.loc[ri + 1:]
        if IGNORE_LABEL is not None:
            shared_counts = ((row == compare) * (row != -1) * (compare != -1)).sum(axis=1)
        else:
            shared_counts = (row == compare).sum(axis=1)

        co_association_matrix[ri, ri + 1:] = shared_counts / n_estimators

    return co_association_matrix


def similarity_to_linkage(similarity_matrix, plot_flag=True, method=LINKAGE_METHOD):
    distance_matrix = 1 - similarity_matrix
    linkage_matrix = hierarchy.linkage(distance_matrix, method=method)

    if plot_flag:
        dendro_row = hierarchy.dendrogram(linkage_matrix, orientation='left')

    return linkage_matrix


def quality(
        ensemble, library_clustering, score=NMI_SCORE,
        replace_noise=REPLACE_NOISE, ignore_label=IGNORE_LABEL
):
    """
    Mean of normalized mutual information.

    Note: Fern and Lin define 'sum' not 'mean' but this must be wrong?
    """
    return np.mean([
        clustering_similarity(
            library_clustering, c, score=score,
            _replace_noise=replace_noise, ignore_label=ignore_label
        )
        for c in ensemble
    ])


def diversity(
        ensemble, score=NMI_SCORE,
        replace_noise=REPLACE_NOISE, ignore_label=IGNORE_LABEL
):
    pairs = list(combinations(ensemble, 2))

    return 1 - np.mean([
        clustering_similarity(
            pair[0], pair[1], score=score,
            _replace_noise=replace_noise, ignore_label=ignore_label
        )
        for pair in pairs
    ])


def criterion(ensemble, library_clustering, alpha=ALPHA):
    return (
        alpha * quality(ensemble, library_clustering)
        + (1 - alpha) * diversity(ensemble)
    )


def build_similarity_graph(ensemble):
    pairwise_nmi = np.zeros([len(ensemble), len(ensemble)])
    NMIG = nx.Graph()  # graph of similarity scores

    for i, e in enumerate(ensemble):
        NMIG.add_node(i)

    pairs = list(combinations(range(len(ensemble)), 2))

    for pi, pair in enumerate(pairs):

        if pi % 10000 == 0:
            print(pi)

        _weight = clustering_similarity(
            ensemble[pair[0]],
            ensemble[pair[1]],
            score=NMI_SCORE,
            _replace_noise=REPLACE_NOISE,
            ignore_label=IGNORE_LABEL
        )

        NMIG.add_edge(
            u_of_edge=pair[0],
            v_of_edge=pair[1],
            weight=_weight
        )

        pairwise_nmi[pair[0], pair[1]] = _weight

    return NMIG, pairwise_nmi


def lables_to_partition(labels):
    partition = {
        i: []
        for i in np.unique(labels)
    }
    for i, j in enumerate(labels):
        partition[j].append(i)

    return list(partition.values())


def build_ensemble(library, library_clustering, k=ENSEMBLE_SIZE):
    ensemble_indices = []
    ensemble = []

    # first we select the highest quality clustering
    all_quality = [
        quality([c], library_clustering)
        for c in library
    ]
    best_index = np.argmax(all_quality)

    ensemble_indices.append(best_index)
    ensemble.append(library[best_index])

    # now we greedily add the clustering that maximises the criterion
    old_best = 0
    while True:

        new_scores = [
            criterion(ensemble + [c], library_clustering)
            if i not in ensemble_indices
            else -1
            for i, c in enumerate(library)
        ]
        best_index = np.argmax(new_scores)
        new_best = new_scores[best_index]

        old_best = new_best
        ensemble_indices.append(best_index)
        ensemble.append(library[best_index])

        if len(ensemble) >= k:
            break

    return ensemble, ensemble_indices


def build_cas_ensemble(partition, library, library_clusters=None, select_best=True):
    ensemble_indices = []
    ensemble = []

    for com in partition:

        if select_best and library_clusters is not None:
            index = np.argmax([
                quality([library[c]], library_clusters)
                for c in com
            ])
        else:
            index = np.random.choice(list(com))

        ensemble_indices.append(index)
        ensemble.append(
            library[index]
        )

    return ensemble, ensemble_indices

ensemble_outputs = {}

for r in range(N_REPEATS):

    print("Repeat: ", r)
    np.random.seed(BASE_SEED + r)

    if CLUSTERING_ALGO == 'kmeans':
        all_results = {
            run_id: sample_results(load_results(run_id))
            for run_id in run_metadata[SEARCH_TYPE].keys()
        }
    elif CLUSTERING_ALGO == 'hdbscan':
        all_results = {
            run_id: sample_results(filter_results(load_results(run_id)))
            for run_id in run_metadata[SEARCH_TYPE].keys()
        }

    library = all_results[RUN_IDS_TO_INCLUDE[0]]
    for run_id in RUN_IDS_TO_INCLUDE[1:]:
        library = pd.concat(
            [library, all_results[run_id]], ignore_index=True
        )

    library_co_association_matrix = ensemble_to_co_association(library)
    library_linkage_matrix = similarity_to_linkage(library_co_association_matrix)
    library_clusters = hierarchy.fcluster(library_linkage_matrix, t=LIBRARY_N_CLUSTER, criterion='maxclust')

    if ENSEMBLE_SELECTION_METHOD == 'JC':
        ensemble, e_indices = build_ensemble(library.labels, library_clusters)

    elif ENSEMBLE_SELECTION_METHOD == 'CAS':
        NMIG, pairwise_nmi = build_similarity_graph(library.labels)
        adj_mat = nx.to_numpy_array(NMIG)
        sc = SpectralClustering(ENSEMBLE_SIZE, affinity='precomputed', n_init=100)
        sc.fit(adj_mat)
        sc_coms = lables_to_partition(sc.labels_)
        ensemble, e_indices = build_cas_ensemble(sc_coms, library.labels, library_clusters=library_clusters)

    final_co_association_matrix = ensemble_to_co_association(ensemble)
    final_linkage_matrix = similarity_to_linkage(final_co_association_matrix, plot_flag=False)
    final_clusters = [
        hierarchy.fcluster(final_linkage_matrix, t=nc+1, criterion='maxclust')
        for nc in range(MAXCLUST)
    ]

    if r == 0:
        base_clusters = final_clusters
        base_library_clusters = library_clusters
        for nc in range(MAXCLUST):
            cs = build_cluster_summary(all_data, final_clusters[nc])
            cs.to_csv(save_dir / ('cluster_summary_nc_%d.csv' % nc), sep=';')

    ensemble_outputs[r] = {
        'seed': BASE_SEED + r,
        'library_clusters': library_clusters,
        'final_clusters': final_clusters,
        'library': library,
        'ensemble_indices': e_indices,
        'ari_with_base_final': [
            adjusted_rand_score(final_clusters[nc], base_clusters[nc])
            for nc in range(MAXCLUST)
        ],
        'ami_with_base_final': [
            adjusted_mutual_info_score(final_clusters[nc], base_clusters[nc])
            for nc in range(MAXCLUST)
        ],
        'ari_with_base_library': adjusted_rand_score(library_clusters, base_library_clusters),
        'ami_with_base_library': adjusted_mutual_info_score(library_clusters, base_library_clusters),
        'ari_with_tessa': [
            adjusted_rand_score(final_clusters[nc], tessa.cluster)
            for nc in range(MAXCLUST)
        ],
        'ami_with_tessa': [
            adjusted_mutual_info_score(final_clusters[nc], tessa.cluster)
            for nc in range(MAXCLUST)
        ]
    }
    print(ensemble_outputs)

