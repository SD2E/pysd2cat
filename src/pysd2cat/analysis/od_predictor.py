import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from harness.test_harness_class import TestHarness
from harness.th_model_instances.hamed_models.random_forest_regression import random_forest_regression

def main():
    # Reading in data from versioned-datasets repo.
    # Using the versioned-datasets repo is probably what most people want to do, but you can read in your data however you like.
    df = pd.read_csv('data/tx_od_2.csv')

    # list of feature columns to use and/or normalize:
    sparse_cols = ['growth_media_1', 'growth_media_2', \
                    'inc_temp', 'inc_time_1', 'inc_time_2', 'input', 'media',  'output', \
                    'plate_id',  'post_well',\
                     'pre_well', 'replicate','glycerol_plate_index', \
                    'source_container', 'strain', 'strain_circuit', 'well','glycerol_stock',]

    continuous_cols = ['od']
    feature_cols = sparse_cols + continuous_cols
    print(feature_cols)
    cols_to_predict = ['post_od_raw']

    train1, test1 = train_test_split(df, test_size=0.2, random_state=5)

    # TestHarness usage starts here, all code before this was just data input and pre-processing.

    # Here I set the path to the directory in which I want my results to go by setting the output_location argument. When the Test Harness
    # runs, it will create a "test_harness_results" folder inside of output_location and place all results inside the "test_harness_results"
    # folder. If the "test_harness_results" folder already exists, then previous results/leaderboards will be updated.
    # In this example script, the output_location has been set to the "examples" folder to separate example results from other ones.
    examples_folder_path = os.getcwd()
    print("initializing TestHarness object with output_location equal to {}".format(examples_folder_path))
    print()
    th = TestHarness(output_location=examples_folder_path)

    th.run_custom(function_that_returns_TH_model=random_forest_regression, dict_of_function_parameters={}, training_data=train1,
                  testing_data=test1, data_and_split_description="od functions with glycerol stock now",
                  cols_to_predict=cols_to_predict,index_cols=['id'],
                  feature_cols_to_use=feature_cols, normalize=True, feature_cols_to_normalize=continuous_cols,
                  feature_extraction="rfpimp_permutation", predict_untested_data=False,sparse_cols_to_use=sparse_cols)



if __name__ == '__main__':
    main()