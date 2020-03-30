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


# TODO: put all data csvs in a data folder
def main():
    # NOTE: The code in this module only needs to be ran once. Do not run it if you have already done so.
    # I went ahead and commented out the to_csv lines just as a precaution (even though there is a random seed).
    # I created this module because splitting takes a while to run,
    # so it saves the splits as csv's so they can be read in by model code in other modules.

    full_df_cleaned = pd.read_csv("full_live_dead_df_cleaned.csv")
    print("Shape of full_df_cleaned: {}".format(full_df_cleaned.shape))

    train_bank, test_df = train_test_split(full_df_cleaned, test_size=0.3, random_state=5,
                                           stratify=full_df_cleaned[['(conc, time)']])
    # train_bank.to_csv("train_bank.csv", index=False)
    # test_df.to_csv("test_df.csv", index=False)

    # the following code takes an equal number of samples from each (conc, time) class to ensure
    # each class has the same number of sample points.

    min_num_events = full_df_cleaned["(conc, time)"].value_counts()[-1]
    balanced_df_cleaned = full_df_cleaned.groupby("(conc, time)"). \
        apply(lambda x: x.sample(min_num_events, random_state=5)).reset_index(drop=True)
    print("Shape of balanced_df_cleaned: {}".format(balanced_df_cleaned.shape))

    balanced_train_bank, balanced_test_df = train_test_split(balanced_df_cleaned, test_size=0.3, random_state=5,
                                                             stratify=balanced_df_cleaned[['(conc, time)']])
    # balanced_train_bank.to_csv("balanced_train_bank.csv", index=False)
    # balanced_test_df.to_csv("balanced_test_df.csv", index=False)


if __name__ == '__main__':
    main()
