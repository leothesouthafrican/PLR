#cluster_comparison.py
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import hdbscan
import numpy as np

def drop_na_values(df1, df2):
    return df1.dropna(), df2.dropna()


def perform_hdbscan(umap_result, min_cluster_size=5):
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    hdbscan_model.fit(umap_result)
    return hdbscan_model.labels_

def calculate_silhouette(umap_result, labels):
    filtered_umap_result = umap_result[labels != -1]
    filtered_labels = labels[labels != -1]
    if len(set(filtered_labels)) > 1:
        return silhouette_score(filtered_umap_result, filtered_labels)
    return float('-inf')

def plot_clusters(umap_result, labels, title):
    filtered_umap_result = umap_result[labels != -1]
    filtered_labels = labels[labels != -1]
    plt.scatter(filtered_umap_result[:, 0], filtered_umap_result[:, 1], c=filtered_labels, cmap='Spectral', s=5)
    plt.colorbar(boundaries=np.arange(len(set(filtered_labels)) + 1) - 0.5).set_ticks(np.arange(len(set(filtered_labels))))
    plt.title(title)