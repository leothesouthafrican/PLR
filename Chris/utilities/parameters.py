from skopt.space import Real, Categorical, Integer
import scipy.stats.distributions as dists
import numpy as np

all_model_parameters ={
    'hdbscan': {
#        'hdbscan__cluster_selection_epsilon': Real(0.0, 1.0),
        'hdbscan__cluster_selection_method': Categorical(['eom', 'leaf']),
        'hdbscan__metric': Categorical(['euclidean', 'manhattan']),
        'hdbscan__min_cluster_size': Integer(2, 100),
        'hdbscan__min_samples': Integer(1, 50)
    },
    'pca': {
        'pca__n_components': Integer(2, 160)
    },
    'kmeans': {
        'kmeans__n_clusters': Integer(2, 20),
        'kmeans__n_init': Integer(1, 10),
        'kmeans__init': Categorical(['k-means++', 'random']),
    },
    'umap': {
        'umap__n_neighbors': Integer(2, 1000),
        'umap__min_dist': Real(0, 1),
        'umap__n_components': Integer(2, 160),
        'umap__metric': Categorical([
            'euclidean', 'manhattan'
                         #'minkowski',
            # 'cosine', 'correlation', 'canberra',
            # 'chebyshev', 'braycurtis'
        ])
    },
    # Not in use yet...
    'parametric_umap': {
        # 'parametric_umap__n_neighbors': Integer(2, 1000),
        # 'parametric_umap__min_dist': Real(0, 1),
        # 'parametric_umap__n_components': Integer(2, 160),
        'parametric_umap__autoencoder_loss': Categorical([True, False]),
        'parametric_umap__n_training_epochs': Integer(1, 50),
        'parametric_umapglobal_correlation_loss_weight': Categorical([True, False])
    }
}

# Different format required for randomizedsearchcv
randomized_search_parameters = {
    'hdbscan': {
#        'hdbscan__cluster_selection_epsilon': dists.uniform(0.0, 1.0),
        'hdbscan__cluster_selection_method': ['eom', 'leaf'],
        'hdbscan__metric': ['euclidean', 'manhattan'],
        'hdbscan__min_cluster_size': dists.randint(2, 101),
        'hdbscan__min_samples': dists.randint(1, 51)
    },
    'pca': {
        'pca__n_components': dists.randint(2, 161),
    },
    'kmeans': {
        'kmeans__n_clusters': dists.randint(2, 21),
        'kmeans__n_init': dists.randint(1, 11),
        'kmeans__init': ['k-means++', 'random'],
    },
    'umap': {
        'umap__n_neighbors': dists.randint(2, 1001),
        'umap__min_dist': dists.uniform(0, 1),
        'umap__n_components': dists.randint(2, 161),
        'umap__metric': [
            'euclidean', 'manhattan'
                         #'minkowski',
            # 'cosine', 'correlation', 'canberra',
            # 'chebyshev', 'braycurtis'
        ]
    },
    'parametric_umap': {
#        'parametric_umap__n_neighbors': Integer(2, 1000),
#        'parametric_umap__min_dist': Real(0, 1),
#        'parametric_umap__n_components': Integer(2, 162),
        'parametric_umap__autoencoder_loss': [True, False],
        'parametric_umap__n_training_epochs': dists.randint(1, 51),
        'parametric_umapglobal_correlation_loss_weight': [True, False]
    }
}

run_configs = {
    1: {
        'run_id': 10,
        'random_seed': 42,
        'dim_reducer': 'umap',
        'clustering_algo': 'kmeans',
        'data_path': '../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        'optimiser_score': 'silhouette',
        'search_iter': 1000000,
        'save_freq': 100
    },
    2: {
        'run_id': 10,
        'random_seed': 42,
        'dim_reducer': 'umap',
        'clustering_algo': 'kmeans',
        'data_path': '../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        'optimiser_score': 'dbcv',
        'search_iter': 1000000,
        'save_freq': 100
    },
    3: {
        'run_id': 10,
        'random_seed': 42,
        'dim_reducer': 'umap',
        'clustering_algo': 'hdbscan',
        'data_path': '../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        'optimiser_score': 'silhouette',
        'search_iter': 1000000,
        'save_freq': 100
    },
    4: {
        'run_id': 10,
        'random_seed': 42,
        'dim_reducer': 'umap',
        'clustering_algo': 'hdbscan',
        'data_path': '../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        'optimiser_score': 'dbcv',
        'search_iter': 1000000,
        'save_freq': 100
    },
    5: {
        'run_id': 10,
        'random_seed': 42,
        'dim_reducer': 'pca',
        'clustering_algo': 'kmeans',
        'data_path': '../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        'optimiser_score': 'silhouette',
        'search_iter': 1000000,
        'save_freq': 100
    },
    6: {
        'run_id': 10,
        'random_seed': 42,
        'dim_reducer': 'pca',
        'clustering_algo': 'kmeans',
        'data_path': '../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        'optimiser_score': 'dbcv',
        'search_iter': 1000000,
        'save_freq': 100
    },
    7: {
        'run_id': 10,
        'random_seed': 42,
        'dim_reducer': 'pca',
        'clustering_algo': 'hdbscan',
        'data_path': '../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        'optimiser_score': 'silhouette',
        'search_iter': 1000000,
        'save_freq': 100
    },
    8: {
        'run_id': 10,
        'random_seed': 42,
        'dim_reducer': 'pca',
        'clustering_algo': 'hdbscan',
        'data_path': '../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        'optimiser_score': 'dbcv',
        'search_iter': 1000000,
        'save_freq': 100
    },
    # Added these using relative_validity metric because dbcv not working reliably...
    # Also added fraction_clustered as alternative (only works with hdbscan due to noise label -1).
    9: {
        'run_id': 10,
        'random_seed': 42,
        'dim_reducer': 'pca',
        'clustering_algo': 'hdbscan',
        'data_path': '../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        'optimiser_score': 'rv',
        'search_iter': 1000000,
        'save_freq': 100
    },
    10: {
        'run_id': 10,
        'random_seed': 42,
        'dim_reducer': 'umap',
        'clustering_algo': 'hdbscan',
        'data_path': '../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        'optimiser_score': 'rv',
        'search_iter': 1000000,
        'save_freq': 100
    },
    11: {
        'run_id': 10,
        'random_seed': 42,
        'dim_reducer': 'pca',
        'clustering_algo': 'hdbscan',
        'data_path': '../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        'optimiser_score': 'fraction_clustered',
        'search_iter': 1000000,
        'save_freq': 100
    },
    12: {
        'run_id': 10,
        'random_seed': 42,
        'dim_reducer': 'umap',
        'clustering_algo': 'hdbscan',
        'data_path': '../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        'optimiser_score': 'fraction_clustered',
        'search_iter': 1000000,
        'save_freq': 100
    },
    # Added for parameteric UMAP:
    13: {
        'run_id': 10,
        'random_seed': 42,
        'dim_reducer': 'parametric_umap',
        'clustering_algo': 'kmeans',
        'data_path': '../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        'optimiser_score': 'silhouette',
        'search_iter': 1000000,
        'save_freq': 10
    },
    14: {
        'run_id': 10,
        'random_seed': 42,
        'dim_reducer': 'parametric_umap',
        'clustering_algo': 'kmeans',
        'data_path': '../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        'optimiser_score': 'dbcv',
        'search_iter': 1000000,
        'save_freq': 10
    },
    15: {
        'run_id': 10,
        'random_seed': 42,
        'dim_reducer': 'parametric_umap',
        'clustering_algo': 'hdbscan',
        'data_path': '../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        'optimiser_score': 'silhouette',
        'search_iter': 1000000,
        'save_freq': 10
    },
    16: {
        'run_id': 10,
        'random_seed': 42,
        'dim_reducer': 'parametric_umap',
        'clustering_algo': 'hdbscan',
        'data_path': '../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        'optimiser_score': 'dbcv',
        'search_iter': 1000000,
        'save_freq': 10
    }
}
