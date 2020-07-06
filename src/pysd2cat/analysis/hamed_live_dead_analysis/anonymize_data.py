import os
import numpy as np
import pandas as pd
from pathlib import Path

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)

data_path = os.path.join(os.getcwd(), "datasets")

ethanol_dict = {0.0: 0, 140.0: 1, 210.0: 2, 280.0: 3, 1120.0: 4}

feature_cols = ["FSC-A", "SSC-A", "BL1-A", "RL1-A", "FSC-H", "SSC-H", "BL1-H", "RL1-H", "FSC-W", "SSC-W", "BL1-W", "RL1-W"]
logged_feature_cols = ["log_{}".format(f) for f in feature_cols]
features_dict = {}
for i, f in enumerate(feature_cols):
    features_dict[f] = "feature_{}".format(i + 1)
for i, f in enumerate(logged_feature_cols):
    features_dict[f] = "log_feature_{}".format(i + 1)

# Get rid of this later?
conc_time_dict = {}
time_points = range(1, 13, 1)  # 12 time points
for conc, anonyconc in ethanol_dict.items():
    for t in time_points:
        conc_time_dict["({}, {})".format(conc, t)] = "({}, {})".format(anonyconc, t)


def anonymize_df(df):
    df = df.copy()
    df.rename(columns={"ethanol": "alcohol"}, inplace=True)
    df.rename(columns=features_dict, inplace=True)
    df["alcohol"] = df["alcohol"].map(ethanol_dict)
    return df


def main():
    normalized_train_bank = pd.read_csv(os.path.join(data_path, "normalized_train_bank.csv"))
    normalized_test_df = pd.read_csv(os.path.join(data_path, "normalized_test_df.csv"))
    print(normalized_train_bank.head())
    print()
    anonymized_train_bank = anonymize_df(normalized_train_bank)
    anonymized_test_df = anonymize_df(normalized_test_df)
    print(anonymized_train_bank.head())
    print()
    print(anonymized_test_df.head())
    print()

    # output csvs:
    # anonymized_train_bank.to_csv(os.path.join(data_path, "anonymized_train_bank.csv"), index=False)
    # anonymized_test_df.to_csv(os.path.join(data_path, "anonymized_test_df.csv"), index=False)


if __name__ == '__main__':
    main()
