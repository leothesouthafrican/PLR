from skopt.space import Real, Categorical, Integer

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
