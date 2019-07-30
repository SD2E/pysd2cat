import os
import ast
import pandas as pd
import math
import pysd2cat.analysis.live_dead_analysis as lda

def train_models_for_stats(experiment_df,
                           out_dir='.',
                           data_dir='data/biofab',
                           fcs_columns = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A', 'FL4-A', 
                                          'FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H', 'FL4-H'],
                           dead_volumes=[980., 570., 370., 250., 170., 105.,  64., 29],
                           live_volume=0.0,
                           strain_column_name='kill_volume'
                           ):
    """
    experiment_df has kill_volume, and stain vs not 
    """

    experiment_id = experiment_df.experiment_id.unique()[0]

    for i in range(0, 5):
        for dead_strain_name in dead_volumes:
            for stain in experiment_df.stain.unique():
#                description=experiment_id+"_"+str(i)+"_"+str(live_strain_name)+ \
#                    "_"+str(dead_strain_name)+"_"+str(stain).replace(" ", "-")
                description={ "experiment_id" : experiment_id,
                               "random_state" : i,
                               "live_volume" : live_volume,
                               "dead_volume" : dead_strain_name,
                               "stain" : stain}
                #print(stain)
                if type(stain) is not str and ( stain is None or math.isnan(stain)):
                    df = experiment_df.loc[(experiment_df['stain'].isna())]
                else:
                    df = experiment_df.loc[(experiment_df['stain'] == stain)]
                print(description)

                if not leader_board_case_exists(out_dir, str(description)):
                    lda.add_live_dead_test_harness(df,
                                               strain_column_name,
                                               live_volume, 
                                               dead_strain_name,
                                               fcs_columns=fcs_columns,
                                               out_dir=out_dir,
                                               description=str(description),
                                               random_state=i,
                                               dry_run=True,
                                               feature_importance=False) 

def train_models_for_prediction(experiment_df, out_dir='.',
                                data_dir='data/biofab',
                                strain_column_name='kill_volume',
                                live_strain_name=0,
                                dead_strain_name=980.,
                                fcs_columns = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A', 'FL4-A', 
                                                'FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H', 'FL4-H'],
                                overwrite=False,
                                combine_stains=False
                                ):
    experiment_id = experiment_df.experiment_id.unique()[0]
    random_state=0

    if combine_stains:
        description={ "experiment_id" : experiment_id,
                          "random_state" : random_state,
                          "live_volume" : live_strain_name,
                          "dead_volume" : dead_strain_name,
                          "stain" : "All",
                          "prediction" : True }
        print(description)

        if overwrite or not leader_board_case_exists(out_dir, str(description)):
            res_df = lda.add_live_dead_test_harness(experiment_df,
                                       strain_column_name,
                                       live_strain_name, 
                                       dead_strain_name,
                                       fcs_columns=fcs_columns,
                                       out_dir=out_dir,
                                       description=str(description),
                                       random_state=random_state,
                                       dry_run=False,
                                       feature_importance=False)
            return res_df
        else:
            return None

    else:
        result_df = pd.DataFrame()
        for stain in experiment_df.stain.unique():
            print(stain)
            description={ "experiment_id" : experiment_id,
                          "random_state" : random_state,
                          "live_volume" : live_strain_name,
                          "dead_volume" : dead_strain_name,
                          "stain" : stain,
                          "prediction" : True }
            if type(stain) is not str  and ( stain is None or math.isnan(stain)):
                df = experiment_df.loc[(experiment_df['stain'].isna())]
            else:
                df = experiment_df.loc[(experiment_df['stain'] == stain)]

            if overwrite or not leader_board_case_exists(out_dir, str(description)):
                print(description)
                res_df = lda.add_live_dead_test_harness(df,
                                           strain_column_name,
                                           live_strain_name, 
                                           dead_strain_name,
                                           fcs_columns=fcs_columns,
                                           out_dir=out_dir,
                                           description=str(description),
                                           random_state=random_state,
                                           dry_run=False,
                                           feature_importance=False)
                result_df = result_df.append(res_df, ignore_index=True)
        return result_df
           



##############################
# Leaderboard extraction code
##############################

## def extract_run(x):
##     description = x['Data and Split Description']
##     run = run = "_".join(description.split('_')[0:2])
##     return run

## def extract_kill(x):
##     if get_stain(x):
##         description = x['Data and Split Description']
##         return float(description.split('_')[-1])
##     else:
##         description = x['Data and Split Description']
##         return float(description.split('_')[-3])        

## def get_random_state(x):
##     return x['Data and Split Description'].split('_')[2]

## def get_stain(x):
##     return 'no_stain' not in x['Data and Split Description']


def extract_lb_attribute(x, attribute):
    #print(x['Data and Split Description'])
    try:
        
        description = ast.literal_eval(x['Data and Split Description'].replace(" nan", " None"))
        if attribute in description:
            return description[attribute]
        else:
            return None
    except Exception as e:
        #print("Cannot parse description, skipping: " + str(x['Data and Split Description']))
        return None

def get_leader_board_df(out_dir, expand_description=True):
    leader_board_path=os.path.join(out_dir, 'test_harness_results/custom_classification_leaderboard.html')
    leader_board = pd.read_html(leader_board_path)[0]
    leader_board = leader_board.sort_values(by=['Date', 'Time'], ascending=True)

    if expand_description:
        attributes = ['experiment_id', 'random_state', 'stain', 'live_volume', 'dead_volume']
        for attribute in attributes:
            leader_board.loc[:, attribute] = leader_board.apply(lambda x: extract_lb_attribute(x, attribute), axis = 1)
    return leader_board

def leader_board_case_exists(out_dir, description):
    leader_board_df = get_leader_board_df(out_dir, expand_description=False)
    return len(leader_board_df.loc[leader_board_df['Data and Split Description'] == description]) > 0
