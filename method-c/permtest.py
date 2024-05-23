
import numpy as np
import pandas as pd 

def perm_test(sympdf, clusters, nperm=1000, alpha=0.05, test_stat = "Euclidean"):
    """
    Performs Monte Carlo permutation test with the Euclidean distance between
    the means as the test statistic.

    Arguments:
    ----------
    sympdf: binary symptom scores
    clusters: cluster labels in the form of a df with index matching sympdf and a column named "cluster"
    nperm: number of permutations. Minimum p-value is 1/nperm
    alpha: significance level, not currently doing anything
    test_stat: test statistic to use, only Euclidean implemented right now
    """
    if test_stat != "Euclidean":
        return "Test statistic not implemented"
    
    groups = clusters['cluster'].unique()
    # create an empty data frame with groups as the index and columns 
    # for the test statistic and p-value
    res_teststat = pd.DataFrame(index = groups, columns = groups)
    res_pval = pd.DataFrame(index = groups, columns = groups)

    # list all pairs from n choose 2 of groups
    pairs = []
    for i in range(len(groups)):
        for j in range(i+1,len(groups)):
            pairs.append([groups[i],groups[j]])

    for pair in pairs:
        pairpval, pairtruedist = perm_test_duo(sympdf, clusters, pair, nperm = nperm, test_stat=test_stat)
        res_teststat.loc[pair[0],pair[1]] = pairtruedist
        res_teststat.loc[pair[1],pair[0]] = pairtruedist
        res_pval.loc[pair[1],pair[0]] = pairpval

    return(res_pval, res_teststat)


def perm_test_duo(sympdf, clusters, pair, nperm, test_stat="Euclidean"):
    """
    Performs Monte Carlo permutation test with the Euclidean distance between
    the means as the test statistic of one pair of symptoms 
    input:
        sympdf - binary symptom scores
        clusters - cluster labels in the form of a df with index matching sympdf and a column named "cluster"
        pair - list of two clusters to compare
        nperm - number of permutations
        test_stat - test statistic to use, only Euclidean implemented right now
    """
    #subset sympdf to the clusters in pair
    sympair = sympdf[clusters['cluster'].isin(pair)]
    cluspair = clusters[clusters['cluster'].isin(pair)]
    truedist = np.linalg.norm(sympair[cluspair['cluster'] == pair[0]].mean() - sympair[cluspair['cluster'] == pair[1]].mean())
    dist = [] 
    for i in range(nperm):
        # shuffle class labels
        shuff = np.random.permutation(cluspair['cluster'])
        # calculate distance between means
        if test_stat == "Euclidean":
            shuffdist = np.linalg.norm(sympair[shuff == pair[0]].mean() - sympair[shuff == pair[1]].mean())
        else: 
            return "Test statistic not implemented"
        # add to list
        dist.append(shuffdist)
    # calculate p-value
    pval = sum(dist > truedist)/nperm
    return pval, truedist