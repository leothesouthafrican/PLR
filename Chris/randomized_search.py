"""
Code for running BayesSearchCV to optimise embedding + shallow classifier.
"""
import pickle
import hdbscan
import time
import sys
import wandb
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from utilities import RandomizedSearch
# from umap.parametric_umap import ParametricUMAP
import umap

from utilities import (
    load_symptom_data,
    dbcv,
    dbcv_minkowski,
    calinski_harabasz,
    silhouette,
    davies_bouldin,
    cluster_count,
    run_configs
)
from utilities import randomized_search_parameters as all_model_parameters

GLOBALS = run_configs[int(sys.argv[1])]


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
        'calinski_harabasz': calinski_harabasz,
        'davies_bouldin': davies_bouldin,
        'dbcv_minkowski': dbcv_minkowski,
        'cluster_count': cluster_count
    }

    model.fit(X)
    labels = model.steps[1][1].labels_
    data = model.steps[0][1].transform(X)

    if score == 'all':
        return_dict = {
            score_name: score_func(data, labels)
            for score_name, score_func in score_dict.items()
        }
        return_dict.update({'labels': labels})
        return return_dict
    else:
        return score_dict[score](data, labels)


all_models = {
    'pca': PCA(random_state=GLOBALS['random_seed']),
    'hdbscan': hdbscan.HDBSCAN(gen_min_span_tree=True, core_dist_n_jobs=1),
    'kmeans': KMeans(random_state=GLOBALS['random_seed']),
    'umap': umap.UMAP(random_state=GLOBALS['random_seed']),
    # 'parametric_umap': ParametricUMAP(random_state=GLOBALS['random_seed'])
}

if __name__ == '__main__':
    df = load_symptom_data(GLOBALS['data_path'])

    pipeline_params = {
        **all_model_parameters[GLOBALS['dim_reducer']],
        **all_model_parameters[GLOBALS['clustering_algo']]
    }
    # print(pipeline_params)

    pipe = Pipeline(
        steps=[
            (GLOBALS['dim_reducer'], all_models[GLOBALS['dim_reducer']]),
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
        scoring=GLOBALS['optimiser_score'],
        n_iter=GLOBALS['search_iter'],
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

    start_time = time.time()
    search.fit(df.to_numpy(), callback=wandb_callback)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    print(search.results_)

    with open('./results/results_%s.pickle' % run_name, 'wb') as out_file:
        pickle.dump(search.results_, out_file)
