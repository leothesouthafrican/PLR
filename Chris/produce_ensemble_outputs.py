"""
This script produces as ensemble average over multiple trained pipelines
by counting the frequency with which each patient occurs in the same cluster,
and then applying community detection on the resulting network.
"""
import pickle
import pickle as pk
import sys
import pandas as pd
import numpy as np
import os
from pathlib import Path
import networkx as nx
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score

from utilities import run_configs, load_symptom_data, modularity

BUILD_GRAPH = False
ENSEMBLE_ID = int(sys.argv[1])
ENSEMBLE_DEFINITIONS = {
    1: {
        'SEARCH_TYPE': 'randomized_search',  # perhaps more complete (uniform) exploration of parameter space.
        'SAMPLE_SIZE': 5,  # number of trained pipelines to sample from each run config
        'SAMPLE_METHOD': 'best',  # best or random sample.
        'SAMPLE_SCORE': 'run_objective',  # score to use to define 'best' if that sample method is in use.
                            # can be 'run_objective' in which case it uses the optimiser score
                            # defined in run_config.
        'RUN_IDS_TO_INCLUDE': [1, 2, 3, 4, 5, 6, 7, 8, 13, 15]
    },
    2: {
        'SEARCH_TYPE': 'randomized_search',  # perhaps more complete (uniform) exploration of parameter space.
        'SAMPLE_SIZE': 10,  # number of trained pipelines to sample from each run config
        'SAMPLE_METHOD': 'best',  # best or random sample.
        'SAMPLE_SCORE': 'run_objective',  # score to use to define 'best' if that sample method is in use.
                            # can be 'run_objective' in which case it uses the optimiser score
                            # defined in run_config.
        'RUN_IDS_TO_INCLUDE': [1, 2, 5, 6, 13, 15]
    },
    3: {
        'SEARCH_TYPE': 'randomized_search',  # perhaps more complete (uniform) exploration of parameter space.
        'SAMPLE_SIZE': 20,  # number of trained pipelines to sample from each run config
        'SAMPLE_METHOD': 'best',  # best or random sample.
        'SAMPLE_SCORE': 'run_objective',  # score to use to define 'best' if that sample method is in use.
                            # can be 'run_objective' in which case it uses the optimiser score
                            # defined in run_config.
        'RUN_IDS_TO_INCLUDE': [1, 2, 5, 6, 13, 15]
    },
    4: {
        'SEARCH_TYPE': 'randomized_search',  # perhaps more complete (uniform) exploration of parameter space.
        'SAMPLE_SIZE': 5,  # number of trained pipelines to sample from each run config
        'SAMPLE_METHOD': 'best',  # best or random sample.
        'SAMPLE_SCORE': 'silhouette',  # score to use to define 'best' if that sample method is in use.
        # can be 'run_objective' in which case it uses the optimiser score
        # defined in run_config.
        'RUN_IDS_TO_INCLUDE': [1, 2, 3, 4, 5, 6, 7, 8, 13, 15]
    },
    5: {
        'SEARCH_TYPE': 'randomized_search',  # perhaps more complete (uniform) exploration of parameter space.
        'SAMPLE_SIZE': 10,  # number of trained pipelines to sample from each run config
        'SAMPLE_METHOD': 'best',  # best or random sample.
        'SAMPLE_SCORE': 'silhouette',  # score to use to define 'best' if that sample method is in use.
        # can be 'run_objective' in which case it uses the optimiser score
        # defined in run_config.
        'RUN_IDS_TO_INCLUDE': [1, 2, 5, 6, 13, 15]
    },
    6: {
        'SEARCH_TYPE': 'randomized_search',  # perhaps more complete (uniform) exploration of parameter space.
        'SAMPLE_SIZE': 20,  # number of trained pipelines to sample from each run config
        'SAMPLE_METHOD': 'best',  # best or random sample.
        'SAMPLE_SCORE': 'silhouette',  # score to use to define 'best' if that sample method is in use.
        # can be 'run_objective' in which case it uses the optimiser score
        # defined in run_config.
        'RUN_IDS_TO_INCLUDE': [1, 2, 5, 6, 13, 15]
    },
    7: {
        'SEARCH_TYPE': 'randomized_search',  # perhaps more complete (uniform) exploration of parameter space.
        'SAMPLE_SIZE': 5,  # number of trained pipelines to sample from each run config
        'SAMPLE_METHOD': 'random',  # best or random sample.
        'SAMPLE_SCORE': 'silhouette',  # score to use to define 'best' if that sample method is in use.
        # can be 'run_objective' in which case it uses the optimiser score
        # defined in run_config.
        'RUN_IDS_TO_INCLUDE': [1, 2, 3, 4, 5, 6, 7, 8, 13, 15]
    },
    8: {
        'SEARCH_TYPE': 'randomized_search',  # perhaps more complete (uniform) exploration of parameter space.
        'SAMPLE_SIZE': 10,  # number of trained pipelines to sample from each run config
        'SAMPLE_METHOD': 'random',  # best or random sample.
        'SAMPLE_SCORE': 'silhouette',  # score to use to define 'best' if that sample method is in use.
        # can be 'run_objective' in which case it uses the optimiser score
        # defined in run_config.
        'RUN_IDS_TO_INCLUDE': [1, 2, 5, 6, 13, 15]
    },
    9: {
        'SEARCH_TYPE': 'randomized_search',  # perhaps more complete (uniform) exploration of parameter space.
        'SAMPLE_SIZE': 20,  # number of trained pipelines to sample from each run config
        'SAMPLE_METHOD': 'random',  # best or random sample.
        'SAMPLE_SCORE': 'silhouette',  # score to use to define 'best' if that sample method is in use.
        # can be 'run_objective' in which case it uses the optimiser score
        # defined in run_config.
        'RUN_IDS_TO_INCLUDE': [1, 2, 5, 6, 13, 15]
    },
    10: {
        'SEARCH_TYPE': 'randomized_search',  # perhaps more complete (uniform) exploration of parameter space.
        'SAMPLE_SIZE': 50,  # number of trained pipelines to sample from each run config
        'SAMPLE_METHOD': 'random',  # best or random sample.
        'SAMPLE_SCORE': 'silhouette',  # score to use to define 'best' if that sample method is in use.
        # can be 'run_objective' in which case it uses the optimiser score
        # defined in run_config.
        'RUN_IDS_TO_INCLUDE': [1]
    },
    11: {
        'SEARCH_TYPE': 'randomized_search',  # perhaps more complete (uniform) exploration of parameter space.
        'SAMPLE_SIZE': 100,  # number of trained pipelines to sample from each run config
        'SAMPLE_METHOD': 'random',  # best or random sample.
        'SAMPLE_SCORE': 'silhouette',  # score to use to define 'best' if that sample method is in use.
        # can be 'run_objective' in which case it uses the optimiser score
        # defined in run_config.
        'RUN_IDS_TO_INCLUDE': [1]
    },
    12: {
        'SEARCH_TYPE': 'randomized_search',  # perhaps more complete (uniform) exploration of parameter space.
        'SAMPLE_SIZE': 1000,  # number of trained pipelines to sample from each run config
        'SAMPLE_METHOD': 'random',  # best or random sample.
        'SAMPLE_SCORE': 'silhouette',  # score to use to define 'best' if that sample method is in use.
        # can be 'run_objective' in which case it uses the optimiser score
        # defined in run_config.
        'RUN_IDS_TO_INCLUDE': [1]
    },
    13: {
        'SEARCH_TYPE': 'randomized_search',  # perhaps more complete (uniform) exploration of parameter space.
        'SAMPLE_SIZE': 30,  # number of trained pipelines to sample from each run config
        'SAMPLE_METHOD': 'best',  # best or random sample.
        'SAMPLE_SCORE': 'silhouette',  # score to use to define 'best' if that sample method is in use.
        # can be 'run_objective' in which case it uses the optimiser score
        # defined in run_config.
        'RUN_IDS_TO_INCLUDE': [13]
    },
    14: {
        'SEARCH_TYPE': 'bayes_search',  # perhaps more complete (uniform) exploration of parameter space.
        'SAMPLE_SIZE': 10,  # number of trained pipelines to sample from each run config
        'SAMPLE_METHOD': 'best',  # best or random sample.
        'SAMPLE_SCORE': 'silhouette',  # score to use to define 'best' if that sample method is in use.
        # can be 'run_objective' in which case it uses the optimiser score
        # defined in run_config.
        'RUN_IDS_TO_INCLUDE': [1, 3, 5, 7]
    },
    15: {
        'SEARCH_TYPE': 'bayes_search',  # perhaps more complete (uniform) exploration of parameter space.
        'SAMPLE_SIZE': 10,  # number of trained pipelines to sample from each run config
        'SAMPLE_METHOD': 'best',  # best or random sample.
        'SAMPLE_SCORE': 'silhouette',  # score to use to define 'best' if that sample method is in use.
        # can be 'run_objective' in which case it uses the optimiser score
        # defined in run_config.
        'RUN_IDS_TO_INCLUDE': [1, 5]
    },
    16: {
        'SEARCH_TYPE': 'bayes_search',  # perhaps more complete (uniform) exploration of parameter space.
        'SAMPLE_SIZE': 50,  # number of trained pipelines to sample from each run config
        'SAMPLE_METHOD': 'best',  # best or random sample.
        'SAMPLE_SCORE': 'silhouette',  # score to use to define 'best' if that sample method is in use.
        # can be 'run_objective' in which case it uses the optimiser score
        # defined in run_config.
        'RUN_IDS_TO_INCLUDE': [1]
    },
}

ENSEMBLE = ENSEMBLE_DEFINITIONS[ENSEMBLE_ID]
save_dir = Path('./ensemble_outputs/ensemble_%d' % ENSEMBLE_ID)
os.makedirs(
    save_dir,
    exist_ok=True
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

for key in run_metadata[ENSEMBLE['SEARCH_TYPE']].keys():
    run_configs[key].update(run_metadata[ENSEMBLE['SEARCH_TYPE']][key])


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


def sample_results(
        _results, sample_size=ENSEMBLE['SAMPLE_SIZE'],
        score=ENSEMBLE['SAMPLE_SCORE'],
        sample_method=ENSEMBLE['SAMPLE_METHOD'],
        run_config=None
):

    if score == 'run_objective':
        score = run_config['optimiser_score']

    if len(_results) >= sample_size:

        if sample_method == 'random':
            return _results.sample(sample_size)
        elif sample_method == 'best':
            return _results.sort_values(
                by=score
                ).iloc[0:sample_size]


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


all_results = {
    run_id: sample_results(
        filter_results(load_results(run_id)),
        run_config=run_configs[run_id]
    )
    for run_id in ENSEMBLE['RUN_IDS_TO_INCLUDE']
}

symptom_data = load_symptom_data(run_configs[1]['data_path'])
all_data = pd.read_csv(run_configs[1]['data_path'], index_col=0)

combined_results = all_results[ENSEMBLE['RUN_IDS_TO_INCLUDE'][0]]
for run_id in ENSEMBLE['RUN_IDS_TO_INCLUDE'][1:]:
    combined_results = pd.concat(
        [combined_results, all_results[run_id]], ignore_index=True
    )

df = pd.DataFrame(combined_results.labels)
df = pd.DataFrame(df['labels'].to_list()).transpose()

if BUILD_GRAPH:
    E = nx.Graph()
    n_estimators = len(combined_results)

    for ri, row in df.iterrows():
        if ri % 500 == 0:
            print(ri)
        compare = df.loc[ri + 1:]
        shared_counts = ((row == compare) * (row != -1) * (compare != -1)).sum(axis=1)

        #     for i, ci in enumerate(range(ri+1, 6031)):
        for i, count in enumerate(shared_counts):

            #         count = shared_counts.iloc[i]
            if count > 0:
                E.add_edge(
                    u_of_edge=symptom_data.index[ri],
                    v_of_edge=symptom_data.index[ri + 1 + i],
                    weight=count / n_estimators
                )

    nx.write_weighted_edgelist(
        E, path=save_dir / 'raw_graph.edgelist'
    )
else:
    E = nx.read_weighted_edgelist(
        save_dir / 'raw_graph.edgelist',
        nodetype=int
    )

tessa = pd.read_csv('../clusterings/tessa/cluster_13_111023.csv')
tessa.rename(columns={'Unnamed: 0': 'index'}, inplace=True)

print("Running Louvain community detection...")
#for gamma in [1.00014, 1.00015, 1.00016, 1.00017, 1.00018, 1.00019]:
#for gamma in [1.0001, 1.00015, 1.0002, 1.00025, 1.0003, 1.00035, 1.0004]:
#for gamma in [1, 1.00005, 1.0001, 1.0005, 1.001]:
#for gamma in [1, 1.005, 1.01, 1.05, 1.1]:
#for gamma in [1.15, 1.2, 1.25, 1.3, 1.35]:
#for gamma in [1, 1.005, 1.01, 1.05, 1.1]:
for gamma in [0.99, 0.9905, 0.991, 0.9915, 0.9920, 0.9925, 0.993]:

    print(gamma)
    #coms = nx.community.louvain_communities(E, seed=42, resolution=gamma)
    coms = nx.community.louvain_communities(E, seed=42, resolution=gamma, backend='cugraph')
    print("%d clusters of size: " % len(coms))
    print([len(c) for c in coms])

    with open(save_dir / ('community_partition_gamma_%.3f.pickle' % gamma), 'wb') as outfile:
        pickle.dump(coms, outfile)

    labels = {c: ci for ci, com in enumerate(coms) for c in com}
    labels = [labels[i] for i in symptom_data.index]
    print(
        "Similarity with Tessa clusters: ",
        adjusted_rand_score(labels, tessa.cluster)
    )

    cs = build_cluster_summary(all_data, labels)
    #print(cs.to_string())
    cs.to_csv(save_dir / ('cluster_summary_gamma_%.3f.csv' % gamma), sep=';')
