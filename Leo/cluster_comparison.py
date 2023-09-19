import umap
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import hdbscan
import numpy as np

# Used to compute the contingency table for two attributes
def drop_na_values(df1, df2):
    return df1.dropna(), df2.dropna()

# Used to compute the contingency table for two attributes
def perform_umap(df, n_neighbors=15, min_dist=0.1, n_components=2):
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
    return umap_model.fit_transform(df)

# Used to compute the contingency table for two attributes
def perform_hdbscan(umap_result, min_cluster_size=5):
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    hdbscan_model.fit(umap_result)
    return hdbscan_model.labels_

# Used to compute the contingency table for two attributes
def calculate_silhouette(umap_result, labels):
    filtered_umap_result = umap_result[labels != -1]
    filtered_labels = labels[labels != -1]
    if len(set(filtered_labels)) > 1:
        return silhouette_score(filtered_umap_result, filtered_labels)
    return float('-inf')

# Used to compute the contingency table for two attributes
def plot_clusters(umap_result, labels, title):
    filtered_umap_result = umap_result[labels != -1]
    filtered_labels = labels[labels != -1]
    plt.scatter(filtered_umap_result[:, 0], filtered_umap_result[:, 1], c=filtered_labels, cmap='Spectral', s=5)
    plt.colorbar(boundaries=np.arange(len(set(filtered_labels)) + 1) - 0.5).set_ticks(np.arange(len(set(filtered_labels))))
    plt.title(title)