import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, MeanShift

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


    organism = "ecoli"


    if organism == "yeast":
        train_df = yeast_train_bank
        test_df = yeast_test_df
        feature_cols = fcols_yeast
    elif organism == "basc":
        train_df = basc_train_bank
        test_df = basc_test_df
        feature_cols = fcols_basc_ecoli
    elif organism == "ecoli":
        train_df = ecoli_train_bank
        test_df = ecoli_test_df
        feature_cols = fcols_basc_ecoli
    else:
        raise NotImplementedError()

    n_clusters = 3
    c_model = KMeans(n_clusters=n_clusters)
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


if __name__ == '__main__':
    main()
