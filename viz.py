import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns

def plot_symp_group_freq(clusters, symp_groups = None, sympdf= None, include_misc = True, mode = "mean", saveloc = None):
    """
    symp_groups: symptoms labeled by which group they're in, default Tessa's in clusterins/tessa/symptom_groups.csv
    clusters: patient IDs in column 1, clusters in column 2
    sympdf: dataframe of symptoms, default it all entries in cleaned_data_SYMPTOMS_9_13_23.csv starting with Symptom_
    """
    if sympdf == None:
        df = pd.read_csv("../data/cleaned_data_SYMPTOMS_9_13_23.csv", index_col=0)
        sympdf = df.loc[:, df.columns.str.startswith('Symptom_')]
    if symp_groups == None:
        symp_groups = pd.read_csv("../clusterings/tessa/symptom_groups.csv")
    if not include_misc:
        symp_groups = symp_groups[symp_groups['group'].isin(['Misc','misc']) == False]

    sympdf['cluster'] = clusters.iloc[:,-1]

    if mode == "mean":
        avg_scores_by_grouping = {}

        for grouping in symp_groups.group.unique():
            # calculat the average score for each symptom in the group for each cluster in 'cluster'
            keepcols = symp_groups.loc[symp_groups.group == grouping, 'symptom'].tolist()
            #print(keepcols)
            keepcols.append('cluster')
            a = sympdf.loc[:, keepcols]
            avg = a.groupby('cluster').mean()
            avg_scores_by_grouping[grouping] = avg.mean(axis=1)
        avgdf  = pd.DataFrame(avg_scores_by_grouping)
        plt.figure(figsize=(20,10))
        sns.heatmap(avgdf.transpose(), cmap='RdBu_r', center=0)
        if saveloc is not None:
            plt.savefig(saveloc)
        return(avgdf)
            #avg = sympdf.loc[:, sympdf.columns.str.startswith(grouping)].groupby(clus).mean()
