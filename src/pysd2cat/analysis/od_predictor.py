import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from harness.test_harness_class import TestHarness
from harness.th_model_instances.hamed_models.random_forest_regression import random_forest_regression
from harness.utils.names import Names

def main():
    # Reading in data from versioned-datasets repo.
    # Using the versioned-datasets repo is probably what most people want to do, but you can read in your data however you like.
    df = pd.read_csv('/Users/meslami/Documents/GitRepos/pysd2cat/src/data/tx_od-5.csv')

    
def predict(df):
    # list of feature columns to use and/or normalize:
    #Do these columns form a a unique entity? If not, we need to define a grouping.
    sparse_cols = ['glycerol_stock','growth_media_1', 'growth_media_2', \
                    'inc_temp', 'inc_time_1', 'inc_time_2', 'post_well', 'pre_well','source_container',\
                     'SynBioHub URI']
    df['glycerol_stock'].fillna('blank',inplace=True)
    continuous_cols = ['od','post_gfp_raw','pre_gfp_raw','pre_od_raw']
    feature_cols = sparse_cols + continuous_cols
    print(feature_cols)
    #cols_to_predict = ['post_od_raw']
    cols_to_predict = ['od']

    train1, test1 = train_test_split(df, test_size=0.2, random_state=5)


    '''
    sub_df = df[feature_cols]
    print(len(sub_df))
    sub_df['cnt']=1
    sub_df_grouped = sub_df.groupby(feature_cols).sum()
    sub_df_grouped.to_csv("/Users/meslami/Documents/GitRepos/pysd2cat/src/data/grouped_df.csv")
    print(len(sub_df_grouped))

    media_options = [ 
        "standard_media",
        "rich_media",
        "slow_media",
        "high_osm_media"
    ]
    media_choice = 0
    gen_exp_msg = {
        "sampling" : False,
        "defaults" : {
            "inducer": "none",
            "inducer_conc": 0.0,
            "inducer_unit": "millimolar",
            "fluor_ex": "488:nanometer",
            "fluor_em": "509:nanometer",
            "growth_media_1": media_options[media_choice], 
            "growth_media_2": media_options[media_choice], 
            "gfp_control": "A11",
            "wt_control": "A10", 
            "inc_time_1": "18:hour", 
            "inc_time_2": "18:hour", 
            "inc_temp": "warm_30",  
            "source_plates": None, 
            "od_cutoff": "0.1",
            "store_growth_plate2" : False
        }
    }
    df_test_all = []
    for strain in df["strain"].unique():
        df_test = pd.DataFrame()
#        df_test['od'] = [0.002, 0.001, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125, 0.00015625, 0.000078125,
#                         0.0000390625, 0.00001953125]
        df_test['post_od_raw'] = df.post_od_raw.unique()
        df_test["growth_media_1"]=gen_exp_msg["defaults"]["growth_media_1"]
        df_test["growth_media_2"]=gen_exp_msg["defaults"]["growth_media_2"]
        df_test["inc_temp"]=gen_exp_msg["defaults"]["inc_temp"]
        df_test["inc_time_1"]=gen_exp_msg["defaults"]["inc_time_1"]
        df_test["inc_time_2"] = gen_exp_msg["defaults"]["inc_time_2"]
        df_test["strain"]=strain
        df_test["id"]=df_test.index
        df_test_all.append(df_test)


    df_test_complete = pd.concat(df_test_all)
    print("complete columns")
    print(df_test_complete.head(5))

    '''



    # TestHarness usage starts here, all code before this was just data input and pre-processing.

    # Here I set the path to the directory in which I want my results to go by setting the output_location argument. When the Test Harness
    # runs, it will create a "test_harness_results" folder inside of output_location and place all results inside the "test_harness_results"
    # folder. If the "test_harness_results" folder already exists, then previous results/leaderboards will be updated.
    # In this example script, the output_location has been set to the "examples" folder to separate example results from other ones.

    examples_folder_path = os.getcwd()
    print("initializing TestHarness object with output_location equal to {}".format(examples_folder_path))
    print()
    th = TestHarness(output_location=examples_folder_path)

    th.run_custom(function_that_returns_TH_model=random_forest_regression,
                  dict_of_function_parameters={},
                  training_data=train1,
                  testing_data=test1,
                  data_and_split_description="Using tx_od-5.csv where Dan fixed a bunch of issues. testing again.",
                  cols_to_predict=cols_to_predict,
                  index_cols=feature_cols+cols_to_predict,
                  feature_cols_to_use=feature_cols,
                  normalize=True,
                  feature_cols_to_normalize=continuous_cols,
                  feature_extraction=Names.RFPIMP_PERMUTATION,
                  predict_untested_data=False,
                  sparse_cols_to_use=sparse_cols)




if __name__ == '__main__':
    main()
