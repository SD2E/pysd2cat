import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, MeanShift
from harness.test_harness_class import TestHarness
from harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification
import matplotlib

matplotlib.use("tkagg")

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)


def main():
    yeast_features_0 = ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W", "BL1-H", "BL1-W", "BL1-A"]
    yeast_features_1 = ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W", "BL1-H", "BL1-W", "BL1-A", "RL1-A", "RL1-W", "RL1-H"]
    basc_ecoli_features_0 = ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W"]
    basc_ecoli_features_1 = ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W", "RL1-A", "RL1-W", "RL1-H"]

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

    # set stain here. 0 = don't use stains, 1 = use stains
    stain = 1
    print("stain = {}".format(stain))

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

    organism = "yeast"

    if organism == "yeast":
        train_df = yeast_train_bank
        test_df = yeast_test_df
        feature_cols = fcols_yeast
    elif organism == "bacillus":
        train_df = basc_train_bank
        test_df = basc_test_df
        feature_cols = fcols_basc_ecoli
    elif organism == "ecoli":
        train_df = ecoli_train_bank
        test_df = ecoli_test_df
        feature_cols = fcols_basc_ecoli
    else:
        raise NotImplementedError()

    n_clusters = 4
    c_model = KMeans(n_clusters=n_clusters, random_state=5)
    c_model.fit(train_df[feature_cols])
    test_df["cluster_preds"] = c_model.predict(test_df[feature_cols])

    normalize = "all"
    frequency_table = pd.crosstab(index=test_df['ethanol'],
                                  columns=test_df['cluster_preds'],
                                  normalize=normalize)
    print(frequency_table)
    print()

    sns.heatmap(frequency_table, cmap="Blues")
    plt.title("{}: Ethanol Concentration vs. {} KMeans Clusters.".format(organism, n_clusters))
    plt.show()

    # TODO: right now I have clustering with a train/test split, and then I'm doing another train/test split on the test set...
    # ---------------------------------- cluster-driven labeling and model run ----------------------------------
    # add labels based on cluster results
    # labeled_df = test_df.copy()
    # labeled_df["label"] = -1  # super-dead
    # labeled_df.loc[labeled_df["cluster_preds"] == 0, "label"] = 0  # dead
    # labeled_df.loc[labeled_df["cluster_preds"] == 3, "label"] = 1  # live
    #
    # print(labeled_df["label"].value_counts(dropna=False))
    # print()
    # labeled_df = labeled_df.loc[labeled_df["label"] != -1]
    # print(labeled_df["label"].value_counts(dropna=False))
    # print()
    # print(labeled_df)
    #
    # labeled_train, labeled_test = train_test_split(labeled_df, train_size=0.7, random_state=5,
    #                                                stratify=labeled_df[["ethanol", "time_point", "label"]])
    # print("Shape of labeled_train: {}".format(labeled_train.shape))
    # print("Shape of labeled_test: {}".format(labeled_test.shape))
    #
    # current_path = os.getcwd()
    # output_path = os.path.join(current_path, "ld_definition_results")
    # print("initializing TestHarness object with output_location equal to {} \n".format(output_path))
    # th = TestHarness(output_location=output_path)
    # col_to_predict = "label"
    # th.run_custom(function_that_returns_TH_model=random_forest_classification, dict_of_function_parameters={},
    #               training_data=labeled_train, testing_data=labeled_test,
    #               data_and_split_description="Cluster-driven labels",
    #               cols_to_predict=col_to_predict, feature_cols_to_use=yeast_features_1,
    #               index_cols=["arbitrary_index", "ethanol", "time_point"], normalize=False, feature_cols_to_normalize=yeast_features_1,
    #               feature_extraction="eli5_permutation", predict_untested_data=False)


if __name__ == '__main__':
    main()
