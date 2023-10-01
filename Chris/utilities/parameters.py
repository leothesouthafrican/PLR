from skopt.space import Real, Categorical, Integer

all_model_parameters ={
    'hdbscan': {
        'hdbscan__cluster_selection_epsilon' : Real(0.0, 100.0),
        'hdbscan__cluster_selection_method' : Categorical(['eom', 'leaf']),
        'hdbscan__metric' : Categorical(['euclidean', 'manhattan']),
        'hdbscan__min_cluster_size':Integer(10, 2000),
        'hdbscan__min_samples': Integer(1,1000)
    },
    'pca': {
        'pca__n_components': Integer(5, 150)
    },
    'kmeans': {
        'kmeans__n_clusters': Integer(2, 20),
        'kmeans__n_init': Integer(1,10),
        'kmeans__init': Categorical(['k-means++', 'random']),
    },
    'umap': {

    }
}
