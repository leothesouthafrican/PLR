"""
Script that download Wandb logs and recomputes scores and labels: labels were not saved correctly for earlier runs!

#TODO: change parameter format for Bayes Search runs.
"""
import os
from pathlib import Path
import wandb
import hdbscan
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
import umap
import pickle as pk
import pandas as pd
import numpy as np
import sys

from utilities import run_configs
from utilities import load_symptom_data
from utilities import (
    dbcv,
    rv,
    dbcv_minkowski,
    calinski_harabasz,
    silhouette,
    davies_bouldin,
    all_model_parameters,
    fraction_clustered,
    cluster_count
)

all_paths = {
    'best_bayes_search_runs_paths': {
        1: "rusty-chris/tune_shallow_clustering/zsm7m6j4",
    },
    # Ranodmized search explores the space more completely, and given that suboptimal clusterings may be preferable, might contain important data...
    'best_randomized_search_runs_paths': {
        1: "rusty-chris/tune_shallow_clustering/vphbjdan",
        5: "rusty-chris/tune_shallow_clustering/sie9j576"
    }
}

run_id = int(sys.argv[1])
run_paths = 'best_randomized_search_runs_paths'
for key in all_paths[run_paths].keys():
    run_configs[key]['best_run_path'] = all_paths[run_paths][key]

run_name = '_'.join(
        [
            run_configs[run_id]['dim_reducer'],
            run_configs[run_id]['clustering_algo'],
            run_configs[run_id]['optimiser_score'],
            'run_%d' % run_configs[run_id]['run_id'],
            run_paths.split('_')[1]
        ]
    )

print(run_name)
save_path = Path('./results/') / run_name
os.makedirs(save_path, exist_ok=False)

SCORE_DICT = {
        'silhouette': silhouette,
        'dbcv': dbcv,
        'rv': rv,
        'calinski_harabasz': calinski_harabasz,
        'davies_bouldin': davies_bouldin,
        'dbcv_minkowski': dbcv_minkowski,
        'fraction_clustered': fraction_clustered,
        'cluster_count': cluster_count
}

api = wandb.Api(timeout=100)
run = api.run(run_configs[run_id]['best_run_path'])
# df = run.history()
df = pd.DataFrame.from_dict(run.scan_history())
# df_sorted = df.sort_values(run_configs[run_id]['optimiser_score'], ascending=False)


def get_params(p_keys, row):

    return {
        p: row['current_params.' + p]
        for p in p_keys
    }

all_models = {
    'pca': PCA(random_state=run_configs[run_id]['random_seed']),
    'hdbscan': hdbscan.HDBSCAN(gen_min_span_tree=True, core_dist_n_jobs=1),
    'kmeans': KMeans(random_state=run_configs[run_id]['random_seed']),
    'umap': umap.UMAP(random_state=run_configs[run_id]['random_seed']),
    # 'parametric_umap': ParametricUMAP(random_state=GLOBALS['random_seed'])
}

pipe = Pipeline(
    steps=[
        (run_configs[run_id]['dim_reducer'], all_models[run_configs[run_id]['dim_reducer']]),
        (run_configs[run_id]['clustering_algo'], all_models[run_configs[run_id]['clustering_algo']])
    ]
)

symptom_data = load_symptom_data(run_configs[run_id]['data_path'])


def cv_score(model, X, score=None, omit_score='rv'):
    """
    If score == 'all' we return a dictionary of all scores, which
    can be logged to wandb on each iteration.

    Otherwise this is intended for use as a scorer in <X>SearchCV methods.
    In that case metric should be fixed to allow comparison across different runs.
    """
    score_dict = SCORE_DICT

    model.fit(X)
    labels = model.steps[1][1].labels_
    data = model.steps[0][1].transform(X)

    if score == 'all':
        return_dict = {
            score_name: score_func(data, labels, model=model)
            for score_name, score_func in score_dict.items()
            if score_name != omit_score
        }
        return_dict.update({'labels': labels})
        return return_dict
    else:
        return score_dict[score](data, labels, model=model)


p_keys = sorted([
    col.split('.')[1] for col in df.columns
    if 'current_params' in col
])

scores = {}
count = 0
for ri, row in df.iterrows():
    print(count, ri)
    params = get_params(p_keys, row)
    pipe.set_params(**params)
    scores[ri] = cv_score(pipe, symptom_data, score='all')
    count += 1


scores = pd.DataFrame.from_dict(scores, orient='index')
print(scores)
print(df.head())
# get best score index for each number of clusters and compute similarity with that clustering for each row:
for cc in scores.cluster_count.unique():

    benchmark_ri = int(
        scores[scores.cluster_count == cc].sort_values(run_configs[run_id]['optimiser_score'], ascending=False).iloc[0].name
    )

    scores['rand_score_with_%d_cc_%d' % (benchmark_ri, cc)] = scores.apply(
        lambda row: adjusted_rand_score(row['labels'], scores.loc[benchmark_ri]['labels']),
        axis=1
    )
    scores['mi_score_with_%d_cc_%d' % (benchmark_ri, cc)] = scores.apply(
        lambda row: adjusted_mutual_info_score(row['labels'], scores.loc[benchmark_ri]['labels']),
        axis=1
    )

with open(save_path / 'scores_and_labels_df.pickle', 'wb') as outfile:
    pk.dump(scores, outfile)
