import pandas as pd
import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt
import os
import re
import seaborn as sns

# Revised function to parse a single result line
def parse_result_line(line, dataset_number):
    pattern = r"Silhouette Score: (.*), Number of Clusters: (\d+), Skew Threshold: (.*), Correlation Threshold: (.*), Hidden Size: (\d+), Latent Dim: (\d+), Learning Rate: (.*), Epochs: (\d+), HDBSCAN Params: ({.*})"
    match = re.match(pattern, line)
    if match:
        return {
            'silhouette_score': float(match.group(1)),
            'num_clusters': int(match.group(2)),
            'skew_threshold': float(match.group(3)),
            'corr_threshold': float(match.group(4)),
            'hidden_size': int(match.group(5)),
            'latent_dim': int(match.group(6)),
            'learning_rate': float(match.group(7)),
            'epochs': int(match.group(8)),
            'hdbscan_params': eval(match.group(9)),
            'dataset': dataset_number
        }
    return None


#Drop skewed features
def drop_skewed_features(df, threshold=0.5):
    """
    Drop features that are skewed towards 0 or 1.
    
    Parameters:
        df (DataFrame): The input DataFrame with binary features.
        threshold (float): The skewness threshold. Features with skewness above this value will be dropped.
        
    Returns:
        DataFrame: A new DataFrame with skewed features removed.
    """
    lines = ["Dropped Skewed Features"]
    is_title = [True]
    
    dropped_features = []
    
    for col in df.columns:
        # Calculate the skewness for each feature
        skewness = df[col].mean()
        
        # Check if the feature is skewed towards 0 or 1
        if skewness > threshold or skewness < (1 - threshold):
            line = f"Dropping {col} with skewness {skewness:.4f}"
            lines.append(line)
            is_title.append(False)
            dropped_features.append(col)
            
    # Drop the skewed features
    df_dropped = df.drop(columns=dropped_features)
    
    return df_dropped

def phi_coefficient(contingency_table):
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return 0
    a = contingency_table.iloc[0, 0]
    b = contingency_table.iloc[0, 1]
    c = contingency_table.iloc[1, 0]
    d = contingency_table.iloc[1, 1]
    numerator = (a * d) - (b * c)
    denominator = np.sqrt((a + b) * (a + c) * (b + d) * (c + d))
    return numerator / denominator if denominator != 0 else 0

def drop_correlated_features(df, threshold=0.4):
    """
    Drop highly correlated features based on phi coefficient.

    Parameters:
        df (DataFrame): The input DataFrame with binary features.
        threshold (float): The phi coefficient threshold. Pairs of features with a phi coefficient above this value will be considered for dropping.

    Returns:
        DataFrame: A new DataFrame with highly correlated features removed.
    """
    # Initialize an empty dataframe to store phi values
    phi_values = pd.DataFrame(index=df.columns, columns=df.columns)

    # Calculate phi values
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 >= col2:  # Avoid duplicate calculations
                continue
            contingency_table = pd.crosstab(df[col1], df[col2])
            phi = phi_coefficient(contingency_table)
            phi_values.loc[col1, col2] = phi
            phi_values.loc[col2, col1] = phi  # Symmetric matrix

    # Convert to float
    phi_values = phi_values.astype(float)

    # Identify columns to drop
    to_drop = set()
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 == col2 or col1 in to_drop or col2 in to_drop:
                continue
            phi_value = abs(phi_values.loc[col1, col2])
            if phi_value > threshold:
                to_drop.add(col1)  # Choose one column from the pair to drop

    # Drop the highly correlated columns
    df_dropped = df.drop(columns=to_drop)

    return df_dropped

def aggregate_columns(data, group_dict):
    aggregated_data = pd.DataFrame()

    # Iterate over each column in the DataFrame
    for col in data.columns:
        if col.startswith('Grouped'):
            # Keep 'Grouped' columns as they are
            aggregated_data[col] = data[col]
        elif col in group_dict:
            # Aggregate columns based on their group
            group_name = group_dict[col]
            if group_name not in aggregated_data:
                aggregated_data[group_name] = data[col]
            else:
                aggregated_data[group_name] += data[col]
        else:
            # For columns not in the group_dict, add them as is
            aggregated_data[col] = data[col]

    # Compute the average for each group
    for group in set(group_dict.values()):
        if group in aggregated_data:
            aggregated_data[group] /= len([col for col in group_dict if group_dict[col] == group])

    return aggregated_data

def plot_cluster_averages(data, cluster_labels):
    # Ensure 'cluster' column is not in the DataFrame
    data = data.copy()
    if 'cluster' in data.columns:
        data = data.drop(columns=['cluster'])

    unique_labels = np.unique(cluster_labels)
    unique_labels = unique_labels[unique_labels != -1]

    # Calculate cluster sizes and cluster averages
    cluster_averages = {}
    cluster_sizes = {}
    for label in unique_labels:
        cluster_data = data[cluster_labels == label]
        cluster_avg = cluster_data.mean().sort_values(ascending=False)
        cluster_averages[label] = cluster_avg
        cluster_sizes[label] = len(cluster_data)

    # Sort clusters by size in descending order
    sorted_labels = sorted(unique_labels, key=lambda label: cluster_sizes[label], reverse=True)

    # Determine the top 10 unique features across all clusters
    all_features = set()
    for averages in cluster_averages.values():
        top_features = averages.head(10).index
        all_features.update(top_features)
    top_features = sorted(list(all_features))

    # Determine the global min and max values for the color scale
    global_min = min(average.min() for average in cluster_averages.values())
    global_max = max(average.max() for average in cluster_averages.values())

    # Number of rows and columns for the subplots
    num_clusters = len(sorted_labels)
    num_columns = 3
    num_rows = np.ceil(num_clusters / num_columns).astype(int)

    plt.figure(figsize=(25, num_rows * 4))

    for i, label in enumerate(sorted_labels):
        cluster_data = pd.DataFrame(index=[f'Cluster {label}'], columns=top_features)
        for feature in top_features:
            cluster_data[feature] = cluster_averages[label].get(feature, np.nan)

        plt.subplot(num_rows, num_columns, i + 1)
        sns.heatmap(cluster_data, annot=True, fmt=".2f", cmap="YlGnBu", vmin=global_min, vmax=global_max)
        plt.title(f"Cluster {label} Feature Values - Samples: {cluster_sizes[label]}")
        plt.ylabel("Feature")
        plt.xlabel("Average Value")

    plt.tight_layout()
    plt.show()
    

def plot_solution_correlations(data, cluster_labels):
    # Ensure 'cluster' column is not in the DataFrame
    data = data.copy()
    if 'cluster' in data.columns:
        data = data.drop(columns=['cluster'])

    # Calculate cluster sizes and cluster averages to determine top features
    cluster_averages = {}
    unique_labels = np.unique(cluster_labels)
    unique_labels = unique_labels[unique_labels != -1]  # Exclude noise cluster
    for label in unique_labels:
        cluster_data = data[cluster_labels == label]
        cluster_avg = cluster_data.mean().sort_values(ascending=False)
        cluster_averages[label] = cluster_avg

    # Determine the top 10 unique features across all clusters
    all_features = set()
    for averages in cluster_averages.values():
        top_features = averages.head(10).index
        all_features.update(top_features)
    top_features = sorted(list(all_features))

    # Filter data to include only the top features
    filtered_data = data[top_features]

    # Calculate correlation matrix
    correlation_matrix = filtered_data.corr()

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Matrix Across All Clusters for Selected Features")
    plt.xlabel("Symptom Group")
    plt.ylabel("Symptom Group")
    plt.show()



