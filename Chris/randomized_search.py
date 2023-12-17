"""
Code for running BayesSearchCV to optimise embedding + shallow classifier.
"""
import os
import pickle
from pathlib import Path

import hdbscan
import time
import sys
import wandb
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer
from utilities import RandomizedSearch
from umap.parametric_umap import ParametricUMAP
import umap
import numpy as np

from utilities import (
    load_symptom_data,
    dbcv,
    dbcv_minkowski,
    calinski_harabasz,
    silhouette,
    davies_bouldin,
    cluster_count,
    run_configs,
    rv,
    fraction_clustered,
    is_jsonable
)
from utilities import randomized_search_parameters as all_model_parameters

GLOBALS = run_configs[int(sys.argv[1])]


def cast_float(x):
    return x.astype(np.double)


def cv_score(model, X, score=GLOBALS['optimiser_score']):
    """
    If score == 'all' we return a dictionary of all scores, which
    can be logged to wandb on each iteration.

    Otherwise this is intended for use as a scorer in <X>SearchCV methods.
    In that case metric should be fixed to allow comparison across different runs.
    """
    score_dict = {
        'silhouette': silhouette,
        'dbcv': dbcv,
        'rv': rv,
        'calinski_harabasz': calinski_harabasz,
        'davies_bouldin': davies_bouldin,
        'dbcv_minkowski': dbcv_minkowski,
        'fraction_clustered': fraction_clustered,
        'cluster_count': cluster_count
    }

    model.fit(X)
    labels = model.steps[2][1].labels_
    data = pipe.steps[1][1].transform(pipe.steps[0][1].transform(X))

    if score == 'all':
        return_dict = {
            score_name: score_func(data, labels, model=model)
            for score_name, score_func in score_dict.items()
        }
        return_dict.update({'labels': labels})

        params = model.get_params()
        keys = list(params.keys())
        for p in keys:
            if not is_jsonable(params[p]):
                params.pop(p, None)
        return_dict.update(params)
        return return_dict
    else:
        return score_dict[score](data, labels, model=model)


all_models = {
    'pca': PCA(random_state=GLOBALS['random_seed']),
    'hdbscan': hdbscan.HDBSCAN(gen_min_span_tree=True, core_dist_n_jobs=1),
    'kmeans': KMeans(random_state=GLOBALS['random_seed']),
    'umap': umap.UMAP(random_state=GLOBALS['random_seed']),
    'parametric_umap': ParametricUMAP(random_state=GLOBALS['random_seed'])#, batch_size=250)
}

if __name__ == '__main__':
    df = load_symptom_data(GLOBALS['data_path'])

    all_results = {}

    pipeline_params = {
        **all_model_parameters[GLOBALS['dim_reducer']],
        **all_model_parameters[GLOBALS['clustering_algo']]
    }
    # print(pipeline_params)

    pipe = Pipeline(
        steps=[
            (GLOBALS['dim_reducer'], all_models[GLOBALS['dim_reducer']]),
            ('cast', FunctionTransformer(cast_float)),
            (GLOBALS['clustering_algo'], all_models[GLOBALS['clustering_algo']])
        ]
    )

    run_name = '_'.join(
            [
                GLOBALS['dim_reducer'],
                GLOBALS['clustering_algo'],
                GLOBALS['optimiser_score'],
                'run_%d' % GLOBALS['run_id']
            ]
        )
    save_path = Path('./results') / run_name
    os.makedirs(save_path, exist_ok=False)

    config = {
        **GLOBALS,
        **pipeline_params
    }
    run = wandb.init(
        name=run_name,
        project='tune_shallow_clustering',
        config=config
    )

    search = RandomizedSearch(
        pipeline=pipe,
        param_distributions=pipeline_params,
        scorer=cv_score,
        scoring=GLOBALS['optimiser_score'],
        n_iter=GLOBALS['search_iter'],
        bootstrap=GLOBALS['bootstrap'],
        symptom_frac=GLOBALS['symptom_frac'],
    )

    def wandb_callback(result, current_params, all_scores):
        iter = len(result['x_iters'])
        print('Iteration %d' % iter)

        log_dict = {
            'best_score': result['fun'],
            'best_params': result['x'],
            'current_params': current_params
        }
        log_dict.update(all_scores)

        run.log(log_dict)
        print(log_dict)

        all_results[iter] = log_dict
        if iter % GLOBALS['save_freq'] == 0:
            with open(save_path / 'all_results.pickle', 'wb') as outfile:
                pickle.dump(all_results, outfile)

    start_time = time.time()
    search.fit(df, callback=wandb_callback)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    print(search.results_)

    with open(save_path / 'search_cv_results_%s.pickle' % run_name, 'wb') as out_file:
        pickle.dump(search.results_, out_file)
