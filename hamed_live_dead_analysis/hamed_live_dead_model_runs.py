import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from harness.test_harness_class import TestHarness
from harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)


def run_models_varying_train_amounts(train_bank, test_df, list_of_train_percents, features_to_use, output_path, col_to_predict, desc):
    print("initializing TestHarness object with output_location equal to {} \n".format(output_path))
    th = TestHarness(output_location=output_path)

    for p in list_of_train_percents:
        if p == 1.0:
            train_df = train_bank.copy()
        else:
            train_df, _ = train_test_split(train_bank, train_size=p, random_state=5,
                                           stratify=train_bank[["ethanol", "time_point"]])
        print("Shape of train_df (p = {}): {}".format(p, train_df.shape))
        print()
        th.run_custom(function_that_returns_TH_model=random_forest_classification, dict_of_function_parameters={},
                      training_data=train_df,
                      testing_data=test_df, data_and_split_description="{}_{}".format(desc, p),
                      cols_to_predict=col_to_predict, feature_cols_to_use=features_to_use,
                      index_cols=["arbitrary_index"], normalize=True, feature_cols_to_normalize=features_to_use,
                      feature_extraction=False,
                      predict_untested_data=False)


def main():
    yeast_train_bank = pd.read_csv("datasets/yeast_normalized_train_bank.csv")
    yeast_test_df = pd.read_csv("datasets/yeast_normalized_test_df.csv")
    print("Shape of yeast_train_bank: {}".format(yeast_train_bank.shape))
    print("Shape of yeast_test_df: {}".format(yeast_test_df.shape))
    basc_train_bank = pd.read_csv("datasets/basc_normalized_train_bank.csv")
    basc_test_df = pd.read_csv("datasets/basc_normalized_test_df.csv")
    print("Shape of basc_train_bank: {}".format(basc_train_bank.shape))
    print("Shape of basc_test_df: {}".format(basc_test_df.shape))
    ecoli_train_bank = pd.read_csv("datasets/ecoli_normalized_train_bank.csv")
    ecoli_test_df = pd.read_csv("datasets/ecoli_normalized_test_df.csv")
    print("Shape of ecoli_train_bank: {}".format(ecoli_train_bank.shape))
    print("Shape of ecoli_test_df: {}".format(ecoli_test_df.shape))
    print()

    yeast_features_0 = ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W", "BL1-H", "BL1-W", "BL1-A"]
    yeast_features_1 = ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W", "BL1-H", "BL1-W", "BL1-A", "RL1-A", "RL1-W", "RL1-H"]
    basc_ecoli_features_0 = ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W"]
    basc_ecoli_features_1 = ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W", "RL1-A", "RL1-W", "RL1-H"]

    # set stain here. 0 = don't use stains, 1 = use stains
    stain = 1

    if stain == 0:
        fcols_yeast = yeast_features_0
        fcols_basc_ecoli = basc_ecoli_features_0
        yeast_train_bank = yeast_train_bank.loc[yeast_train_bank["stain"] == 0]
        yeast_test_df = yeast_test_df.loc[yeast_test_df["stain"] == 0]
        basc_train_bank = basc_train_bank.loc[basc_train_bank["stain"] == 0]
        basc_test_df = basc_test_df.loc[basc_test_df["stain"] == 0]
        ecoli_train_bank = ecoli_train_bank.loc[ecoli_train_bank["stain"] == 0]
        ecoli_test_df = ecoli_test_df.loc[ecoli_test_df["stain"] == 0]
    elif stain == 1:
        print("whoohoo!")
        fcols_yeast = yeast_features_1
        fcols_basc_ecoli = basc_ecoli_features_1
        yeast_train_bank = yeast_train_bank.loc[yeast_train_bank["stain"] == 1]
        yeast_test_df = yeast_test_df.loc[yeast_test_df["stain"] == 1]
        basc_train_bank = basc_train_bank.loc[basc_train_bank["stain"] == 1]
        basc_test_df = basc_test_df.loc[basc_test_df["stain"] == 1]
        ecoli_train_bank = ecoli_train_bank.loc[ecoli_train_bank["stain"] == 1]
        ecoli_test_df = ecoli_test_df.loc[ecoli_test_df["stain"] == 1]
    else:
        raise NotImplementedError()

    # train_percents_to_try = list(np.flip(np.linspace(0.1, 1, 10))) + list(np.flip(np.linspace(0.01, 0.09, 9)))
    # train_percents_to_try = [round(x, 2) for x in train_percents_to_try]
    # train_percents_to_try = list(np.flip(train_percents_to_try))
    # print(train_percents_to_try)
    # print()

    current_path = os.getcwd()
    # ethanol_path = os.path.join(current_path, "ethanol_classes_results")
    #
    # conc_time_path = os.path.join(current_path, "conc_time_results")
    # conc_path = os.path.join(current_path, "conc_results")

    # custom_percents = [0.01, 0.1, 0.50, 1.0]
    # print(custom_percents)

    # run_models_varying_train_amounts(custom_percents, feature_cols_1, conc_time_path, '(conc, time)')
    # run_models_varying_train_amounts(custom_percents, feature_cols_2, conc_time_path, '(conc, time)')
    # run_models_varying_train_amounts(custom_percents, feature_cols_1, conc_path, 'kill_volume')
    # run_models_varying_train_amounts(custom_percents, feature_cols_2, conc_path, 'kill_volume')

    new_path = os.path.join(current_path, "new_results")
    # run_models_varying_train_amounts(yeast_train_bank, yeast_test_df, [0.1], fcols_yeast, new_path, 'ethanol', "yeast")
    # run_models_varying_train_amounts(basc_train_bank, basc_test_df, [1.0], fcols_basc_ecoli, new_path, 'ethanol', "basc")
    # run_models_varying_train_amounts(ecoli_train_bank, ecoli_test_df, [1.0], fcols_basc_ecoli, new_path, 'ethanol', "ecoli")


if __name__ == '__main__':
    main()
