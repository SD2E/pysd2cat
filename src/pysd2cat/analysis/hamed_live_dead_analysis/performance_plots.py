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
    regular_models_path = "log_features_april_23/test_harness_results"
    cross_organism_path = "cross_organism_april_23/test_harness_results"
    stain_variants_path = "stain_variants_results/test_harness_results"

    r_leaderboard = pd.read_html(os.path.join(regular_models_path, "custom_classification_leaderboard.html"))[0]
    c_leaderboard = pd.read_html(os.path.join(cross_organism_path, "custom_classification_leaderboard.html"))[0]
    s_leaderboard = pd.read_html(os.path.join(stain_variants_path, "custom_classification_leaderboard.html"))[0]

    r_leaderboard = r_leaderboard[["Data and Split Description", "Accuracy"]]
    c_leaderboard = c_leaderboard[["Data and Split Description", "Accuracy"]]
    s_leaderboard = s_leaderboard[["Data and Split Description", "Accuracy"]]

    r_leaderboard["Data and Split Description"] = \
        (r_leaderboard["Data and Split Description"].str.split("_1.0", expand=True)[0])
    r_leaderboard["Data and Split Description"] = \
        (r_leaderboard["Data and Split Description"].str.split("_0.1", expand=True)[0])
    c_leaderboard["Data and Split Description"] = \
        (c_leaderboard["Data and Split Description"].str.split("_1.0", expand=True)[0])
    c_leaderboard["Data and Split Description"] = \
        (c_leaderboard["Data and Split Description"].str.split("_0.1", expand=True)[0])
    s_leaderboard["Data and Split Description"] = \
        (s_leaderboard["Data and Split Description"].str.split("_1.0", expand=True)[0])
    s_leaderboard["Data and Split Description"] = \
        (s_leaderboard["Data and Split Description"].str.split("_0.1", expand=True)[0])

    print(r_leaderboard)
    print()
    print(c_leaderboard)
    print()
    print(s_leaderboard)
    print()

    combo = pd.concat([r_leaderboard, c_leaderboard, s_leaderboard])

    combo["train"] = ["yeast_stain", "basc_stain", "ecoli_stain",
                      "basc_stain", "ecoli_stain", "basc_stain", "ecoli_stain", "yeast_stain", "yeast_stain",
                      "yeast_non_stain", "basc_non_stain", "ecoli_non_stain"]

    combo["test"] = ["yeast_stain", "basc_stain", "ecoli_stain",
                     "ecoli_stain", "basc_stain", "yeast_stain", "yeast_stain", "basc_stain", "ecoli_stain",
                     "yeast_stain", "basc_stain", "ecoli_stain"]

    combo_1 = combo.copy()
    combo_1 = combo_1.loc[~combo_1["train"].str.contains("non_stain")]
    combo_1["train"] = combo_1["train"].str.split("_stain", expand=True)[0]
    combo_1["test"] = combo_1["test"].str.split("_stain", expand=True)[0]
    # print(combo_1)
    # sns.barplot(combo_1["train"], combo_1["Accuracy"], hue=combo_1["test"])
    # plt.ylabel("Accuracy")
    # plt.xlabel("Training Data")
    # plt.legend(title="Testing Data")
    # plt.title("Accuracy of Random Forest Models in Predicting Ethanol Concentration")
    # plt.show()

    print()

    combo_2 = combo.copy()
    combo_2 = combo_2.loc[~combo_2["Data and Split Description"].str.contains("train_")]
    combo_2["organism"] = combo_2["test"].str.split("_", 1, expand=True)[0]
    combo_2["train"] = combo_2["train"].str.split("_", 1, expand=True)[1]
    print(combo_2)

    sns.barplot(combo_2["organism"], combo_2["Accuracy"], hue=combo_2["train"])
    plt.ylabel("Accuracy")
    plt.xlabel("Organism")
    plt.legend(title="Training Data")
    plt.title("Model Accuracy in Predicting Ethanol Concentration in Stained Samples")
    plt.show()


if __name__ == '__main__':
    main()
