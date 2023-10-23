from skopt.space import Real, Categorical, Integer
import numpy as np

all_model_parameters ={
    'hdbscan': {
        'hdbscan__cluster_selection_epsilon': Real(0.0, 1000.0),
        'hdbscan__cluster_selection_method': Categorical(['eom', 'leaf']),
        'hdbscan__metric': Categorical(['euclidean', 'manhattan']),
        'hdbscan__min_cluster_size': Integer(5, 1000),
        'hdbscan__min_samples': Integer(1, 1000)
    },
    'pca': {
        'pca__n_components': Integer(5, 150)
    },
    'kmeans': {
        'kmeans__n_clusters': Integer(2, 20),
        'kmeans__n_init': Integer(1, 10),
        'kmeans__init': Categorical(['k-means++', 'random']),
    },
    'umap': {
        'umap__n_neighbors': Integer(2, 1000),
        'umap__min_dist': Real(0, 1),
        'umap__n_components': Integer(2, 162),
        'umap__metric': Categorical([
            'euclidean', 'manhattan'
                         #'minkowski',
            # 'cosine', 'correlation', 'canberra',
            # 'chebyshev', 'braycurtis'
        ])
    },
    # Noe in use yet...
    'parametric_umap': {
        'parametric_umap__n_neighbors': Integer(2, 1000),
        'parametric_umap__min_dist': Real(0, 1),
        'parametric_umap__n_components': Integer(2, 162),
        'parametric_umap__metric': Categorical([
            'euclidean', 'manhattan', 'minkowski',
            'cosine', 'correlation', 'canberra',
            'chebyshev', 'braycurtis', 'haversine'
        ]),
        'parametric_umap__autoencoder_loss': Categorical([True, False]),
        'parametric_umap__n_training_epochs': Integer(1, 500),
        'parametric_umap__autocencoder_loss': Categorical([True, False])
    }
}

# Different format required for randomizedsearchcv
randomized_search_parameters = {
    'hdbscan': {
        'hdbscan__cluster_selection_epsilon': np.linspace(0.0, 1000.0, num=1000000),
        'hdbscan__cluster_selection_method': ['eom', 'leaf'],
        'hdbscan__metric': ['euclidean', 'manhattan'],
        'hdbscan__min_cluster_size': [i for i in range(1001) if i > 5],
        'hdbscan__min_samples': [i for i in range(1001)]
    },
    'pca': {
        'pca__n_components': [i for i in range(151) if i > 2]
    },
    'kmeans': {
        'kmeans__n_clusters': [i for i in range(21) if i > 1],
        'kmeans__n_init': [i for i in range(11)],
        'kmeans__init': ['k-means++', 'random'],
    },
    'umap': {
        'umap__n_neighbors': [i for i in range(1001) if i > 1],
        'umap__min_dist': np.linspace(0, 1, num=10000),
        'umap__n_components': [i for i in range(163) if i > 1],
        'umap__metric': [
            'euclidean', 'manhattan'
                         #'minkowski',
            # 'cosine', 'correlation', 'canberra',
            # 'chebyshev', 'braycurtis'
        ]
    }
}

run_configs = {
    1: {
        'run_id': 0,
        'random_seed': 42,
        'dim_reducer': 'umap',
        'clustering_algo': 'kmeans',
        'data_path': '../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        'optimiser_score': 'silhouette',
        'search_iter': 1000000
    },
    2: {
        'run_id': 0,
        'random_seed': 42,
        'dim_reducer': 'umap',
        'clustering_algo': 'kmeans',
        'data_path': '../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        'optimiser_score': 'dbcv',
        'search_iter': 1000000
    },
    3: {
        'run_id': 0,
        'random_seed': 42,
        'dim_reducer': 'umap',
        'clustering_algo': 'hdbscan',
        'data_path': '../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        'optimiser_score': 'silhouette',
        'search_iter': 1000000
    },
    4: {
        'run_id': 0,
        'random_seed': 42,
        'dim_reducer': 'umap',
        'clustering_algo': 'hdbscan',
        'data_path': '../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        'optimiser_score': 'dbcv',
        'search_iter': 1000000
    },
    5: {
        'run_id': 0,
        'random_seed': 42,
        'dim_reducer': 'pca',
        'clustering_algo': 'kmeans',
        'data_path': '../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        'optimiser_score': 'silhouette',
        'search_iter': 1000000
    },
    6: {
        'run_id': 0,
        'random_seed': 42,
        'dim_reducer': 'pca',
        'clustering_algo': 'kmeans',
        'data_path': '../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        'optimiser_score': 'dbcv',
        'search_iter': 1000000
    },
    7: {
        'run_id': 0,
        'random_seed': 42,
        'dim_reducer': 'pca',
        'clustering_algo': 'hdbscan',
        'data_path': '../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        'optimiser_score': 'silhouette',
        'search_iter': 1000000
    },
    8: {
        'run_id': 0,
        'random_seed': 42,
        'dim_reducer': 'pca',
        'clustering_algo': 'hdbscan',
        'data_path': '../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        'optimiser_score': 'dbcv',
        'search_iter': 1000000
    },
}