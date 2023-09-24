# data_loading.py

import pandas as pd

def load_data():
    raw_data = pd.read_csv('/Users/leo/Programming/PLR/Leo/data/cleaned_data_SYMPTOMS_9_13_23_DNA.csv')
    data_symp_groups = pd.read_csv('/Users/leo/Programming/PLR/Leo/data/skew_corr_groupadd.csv', usecols=['Grouped_Neuro_Sensory', 'Grouped_Cognitive_Memory', 'Grouped_Gastrointestinal', 'Grouped_Respiratory_Cardiac', 'Grouped_Eye_Vision'])
    data_symp_groups_all = pd.read_csv('/Users/leo/Programming/PLR/Leo/data/skew_corr_groupadd.csv')
    demographics = pd.read_csv('/Users/leo/Programming/PLR/Leo/data/non_binary_data_processed.csv')
    
    demo_all = pd.concat([demographics, data_symp_groups_all], axis=1).drop(['Unnamed: 0'], axis=1)
    
    dataset = {'data_symp_groups_all': demo_all}
    
    return dataset
