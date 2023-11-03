import networkx as nx
from itertools import combinations
import pandas as pd
from scipy.spatial import distance
import pickle
import time


def load_symptom_data(data_path):
    df = pd.read_csv(data_path, index_col=0)
    df = df.fillna(value=0)

    symptoms = [
        col
        for col in df.columns
        if 'Symptom' in col
    ]
    return df[symptoms]


def pairwise_weight(v1, v2):
    """
    Weight metric to be used in a Networkx weighted graph.

    In general should be inverse-distance-like i.e. more closely related
    vectors have a higher weight.

    This version just counts number of shared elements in binary vectors.
    Args:
        v1: numpy.array
        v2: numpy.array
            Vectors to compute the weight between

    Returns:
         float: weight
    """
    return (v1 * v2).sum()


def pairwise_relative_weight(v1, v2):
    """
    Weight metric to be used in a Networkx weighted graph.

    This version computes the number of shared elements in binary vectors,
    divided by the sum of the larger two vectors.

    This is taking the number of shared symptoms relative to the symptom count
    of the patient with the most symptoms.

    Args:
        v1: numpy.array
        v2: numpy.array
            Vectors to compute the weight between

    Returns:
         float: weight
    """
    return (v1 * v2).sum() / max(v1.sum(), v2.sum())


def pairwise_hamming_complement(v1, v2):
    """
    Weight metric to be used in a Networkx weighted graph.

    This version uses 1 minus the hamming distance.

    Args:
        v1: numpy.array
        v2: numpy.array
            Vectors to compute the weight between

    Returns:
         float: weight
    """

    return 1 - distance.hamming(v1, v2)


def build_graph(
        df, weight_metric=pairwise_weight, threshold=0.0,
        verbose=True,
        save_path=None
):
    """
    Method builds graph from a dataframe df where each
    column is to be a node, and the edge weight is computed between each
    pair of column vectors using the weight_metric.

        Parameters:
            df: pd.DataFrame
                Data to build graph from - columns are nodes.
            weight_metric: function
                Function that accepts two vectors as args and returns
                a real-valued weight, where a higher number means the
                two vectors are 'closer' in some sense.
            threshold: float
                Only include edges above this threshold weight.
                Pruning may improve algorithm performance e.g. for community detection.
            save_path: Path
                Where to save the network.
            verbose: bool
                Whether to print progress.

        Returns:
            G: nx.Graph
                Undirected weighted graph.
    """
    start_time = time.time()
    G = nx.Graph()
    for col in df.columns:
        G.add_node(col)

    pairs = list(combinations(df.columns, 2))
    for pi, pair in enumerate(pairs):

        if verbose and pi % 100000 == 0:
            print("Pair %d of %d." % (pi, len(pairs)))
            print("Elapsed time: %.2f" % ((time.time() - start_time) / 60))

        _weight = weight_metric(df[pair[0]], df[pair[1]])
        if _weight > threshold:
            G.add_edge(
                u_of_edge=pair[0], v_of_edge=pair[1],
                weight=_weight
            )

    if save_path is not None:
        nx.write_weighted_edgelist(G, path=save_path)
    else:
        return G


def build_patient_graph(
        data_path='../../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        save_path='../graphs/full_patient_graph_shared_symptom_count.edgelist',
        weight_metric=pairwise_weight,
        threshold=0.0
):
    df = load_symptom_data(data_path)
    return build_graph(
        df.transpose(),
        save_path=save_path,
        weight_metric=weight_metric,
        threshold=threshold
    )


def modularity(partition, graph=None, nodelist=None, gamma=1):
    """
    Function computes the modularity value for a certain partition of the patients.

    Can be computed for a subset of the dataset by passing nodelist.

    Args:
        partition: iterable of sets of patient ids
            Representing the clusters to be tested.
            e.g. format partition=[{3,5,10}, {6, 12, ...}, {...}]

        graph: (optional) networkx.Graph
            Graph on which to compute modularity.
            If not provided, graph will be read from local pre-computed patient graph.
            Note: this is still slow because it is a big graph!

        nodelist: (optional) list of patient ids
            Corresponding to the subset of the dataset that you want to test.

        gamma: resolution parameter (default: 1)
            If resolution is less than 1, modularity favors larger communities.
            Greater than 1 favors smaller communities.
            We can use this to test for structure at different scales (see modularity_sweep method)

    Returns:

    """
    if graph is None:
        G = nx.read_gml('./graphs/full_patient_graph_shared_symptom_count.gml.gz')
    else:
        G = graph

    if nodelist is not None:

        try:
            assert set(nodelist) in set(G.nodes)
        except AssertionError as e:
            print("nodelist parameter must contain valid patient id number (from the index column of main data).")
            raise e
    else:
        nodelist = list(G.nodes)

    try:
        assert set(nodelist) == set().union(*partition)
    except AssertionError as e:
        print("partition parameter must contain all patient id numbers")
        raise e

    return nx.community.modularity(G, partition, resolution=gamma)


# TODO: implement sweep
def modularity_sweep(partition, nodelist=None):
    """
    Sweep gamma values to test modularity at different scales.
    """
    pass


def load_communities(path='../graphs/full_patient_graph_louvain_communities_gamma_1.pickle'):
    with open(path, 'rb') as infile:
        communities = pickle.load(infile)

    return communities
