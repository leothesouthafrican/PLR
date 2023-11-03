from graph_utilities import build_patient_graph, pairwise_relative_weight, pairwise_hamming_complement

# First build default graph, which just counts shared symptoms:
# build_patient_graph()

# Next build the graph of patients with edges weighted by relative symptom count:
# build_patient_graph(
#         data_path='../../data/cleaned_data_SYMPTOMS_9_13_23.csv',
#         save_path='../graphs/full_patient_graph_shared_symptom_relative.edgelist',
#         weight_metric=pairwise_relative_weight,
#         threshold=0.1
# )
# Note: we currently use a relatively conservative threshold of 10% shared symptoms.

# Next build the graph of patients with edges weighted by pairwise_hamming_complement:
build_patient_graph(
        data_path='../../data/cleaned_data_SYMPTOMS_9_13_23.csv',
        save_path='../graphs/full_patient_graph_shared_symptom_hamming_complement_threshold_0.5.edgelist',
        weight_metric=pairwise_hamming_complement,
        threshold=0.5
)
# Note: trying a higher threshold - need to explore all systematically.

