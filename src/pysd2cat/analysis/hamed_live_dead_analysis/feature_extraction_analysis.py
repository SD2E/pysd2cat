import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from six import string_types

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)
CWD = os.getcwd()


def main():
    th_results_path = "log_features_april_23/test_harness_results"
    leaderboard = pd.read_html(os.path.join(th_results_path, "custom_classification_leaderboard.html"))[0]

    yeast_id = leaderboard.loc[leaderboard["Data and Split Description"].str.contains("yeast")]["Run ID"].iloc[0]
    basc_id = leaderboard.loc[leaderboard["Data and Split Description"].str.contains("bacillus")]["Run ID"].iloc[0]
    ecoli_id = leaderboard.loc[leaderboard["Data and Split Description"].str.contains("ecoli")]["Run ID"].iloc[0]
    yeast_feature_importances = pd.read_csv(os.path.join(th_results_path,
                                                         "runs/run_{}/feature_importances.csv".format(yeast_id)))
    basc_feature_importances = pd.read_csv(os.path.join(th_results_path,
                                                        "runs/run_{}/feature_importances.csv".format(basc_id)))
    ecoli_feature_importances = pd.read_csv(os.path.join(th_results_path,
                                                         "runs/run_{}/feature_importances.csv".format(ecoli_id)))
    yeast_feature_importances["organism"] = "yeast"
    basc_feature_importances["organism"] = "bacillus"
    ecoli_feature_importances["organism"] = "ecoli"

    feature_importances = pd.concat([yeast_feature_importances,
                                     basc_feature_importances,
                                     ecoli_feature_importances])

    # print(feature_importances)
    # -----------------------
    #
    # feature_importances = feature_importances.loc[feature_importances["organism"] == "yeast"]
    # feature_cols = ["FSC-A", "SSC-A", "BL1-A", "RL1-A", "FSC-H", "SSC-H", "BL1-H", "RL1-H", "FSC-W", "SSC-W", "BL1-W", "RL1-W"]
    # logged_feature_cols = ["log_{}".format(f) for f in feature_cols]
    # features_dict = {}
    # for i, f in enumerate(logged_feature_cols):
    #     features_dict[f] = "feature_{}".format(i + 1)
    # feature_importances["Feature"] = feature_importances["Feature"].map(features_dict)
    # print(feature_importances)

    # -----------------------

    importance_method = leaderboard["Feature Extraction"][0]
    print(importance_method)

    sns.barplot(feature_importances["Feature"], feature_importances["Importance"], hue=feature_importances["organism"])
    plt.ylabel("Importance ({})".format(importance_method))
    plt.title("Feature Importance When Predicting Alcohol Concentration")
    plt.show()


if __name__ == '__main__':
    main()
