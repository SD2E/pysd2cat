import numpy as np

def get_per_experiment_statistics(df):
    """
    Convert sample data in df into per experiment statistics.
    """
    experiment_groups = df.groupby(['experiment_id'])
    experiment_groups_result = \
        experiment_groups.agg({'pre_OD600': [np.mean, np.std], 
                               'OD600' : [np.mean, np.std]}).reset_index()
    return experiment_groups_result

def get_per_experiment_statistics_by_od(df):
    """
    Convert sample data in df into per experiment by od statistics.
    """
    experiment_od_groups = df.groupby(['experiment_id', 'od'])
    experiment_od_groups_result = \
        experiment_od_groups.agg({'pre_OD600': [np.mean, np.std], 
                                  'OD600' : [np.mean, np.std]}).reset_index()
    return experiment_od_groups_result

def get_strain_statistics_by_od(df):
    groups = df.groupby(['strain', 'od', 'strain_circuit'])
    result = groups.agg({'pre_OD600': [np.mean, np.std], 
                         'OD600' : [np.mean, np.std]}).reset_index()
    return result