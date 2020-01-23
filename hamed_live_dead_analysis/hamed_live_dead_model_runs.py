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


def main():
    full_df_cleaned = pd.read_csv("full_live_dead_df_cleaned.csv")
    print("Shape of full_df_cleaned: {}".format(full_df_cleaned.shape))

    feature_cols_1 = ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W", "BL1-H", "BL1-W", "BL1-A", "RL1-H", "RL1-W", "RL1-H"]
    feature_cols_2 = ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W", "BL1-H", "BL1-W", "BL1-A"]

    # The following is commented out because it only needed to be ran once.
    # I did this because splitting takes a while to run,
    # so I saved the splits as csv's and they are read them in each time the code is run.
    # train_bank, test_df = train_test_split(full_df_cleaned, test_size=0.3, random_state=5, stratify=full_df_cleaned[['(conc, time)']])
    # train_bank.to_csv("train_bank.csv", index=False)
    # test_df.to_csv("test_df.csv", index=False)

    train_bank = pd.read_csv("train_bank.csv")
    test_df = pd.read_csv("test_df.csv")
    print("Shape of train_bank: {}".format(train_bank.shape))
    print("Shape of test_df: {}".format(test_df.shape))

    train_percents_to_try = list(np.flip(np.linspace(0.1, 1, 10))) + list(np.flip(np.linspace(0.01, 0.09, 9)))
    train_percents_to_try = [round(x, 2) for x in train_percents_to_try]
    train_percents_to_try = list(np.flip(train_percents_to_try))
    print(train_percents_to_try)
    print()

    current_path = os.getcwd()
    print("initializing TestHarness object with output_location equal to {}".format(current_path))
    print()
    th = TestHarness(output_location=current_path)

    def run_models_varying_train_amounts(list_of_train_percents, features_to_use):
        for p in list_of_train_percents:
            if p == 1.0:
                train_df = train_bank.copy()
            else:
                train_df, _ = train_test_split(train_bank, train_size=p, random_state=5, stratify=train_bank[['(conc, time)']])
            print("Shape of train_df (p = {}): {}".format(p, train_df.shape))
            print()
            th.run_custom(function_that_returns_TH_model=random_forest_classification, dict_of_function_parameters={},
                          training_data=train_df,
                          testing_data=test_df, data_and_split_description="{}".format(p),
                          cols_to_predict='(conc, time)', feature_cols_to_use=features_to_use,
                          index_cols=["arbitrary_index"], normalize=True, feature_cols_to_normalize=features_to_use,
                          feature_extraction=False,
                          predict_untested_data=False)

    percents_1_and_40 = [0.01, 0.40]
    print(percents_1_and_40)
    run_models_varying_train_amounts(percents_1_and_40, feature_cols_2)


if __name__ == '__main__':
    main()
