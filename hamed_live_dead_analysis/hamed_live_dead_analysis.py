import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import confusion_matrix
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)


def conf_matrix_and_clustermap(leaderboard, rl1s=True, percent_train_data=0.01,
                               normalize='true', cm_labels=None, th_results_path=os.getcwd(),
                               cbar=True):
    if rl1s:
        rl1s = "RL1s Used"
    else:
        rl1s = "RL1s Not Used"

    if cm_labels is None:
        confusion_matrix_labels = ["(0.0, 1)", "(140.0, 1)", "(210.0, 1)", "(280.0, 1)", "(1120.0, 1)",
                                   "(0.0, 2)", "(140.0, 2)", "(210.0, 2)", "(280.0, 2)", "(1120.0, 2)",
                                   "(0.0, 3)", "(140.0, 3)", "(210.0, 3)", "(280.0, 3)", "(1120.0, 3)",
                                   "(0.0, 4)", "(140.0, 4)", "(210.0, 4)", "(280.0, 4)", "(1120.0, 4)",
                                   "(0.0, 5)", "(140.0, 5)", "(210.0, 5)", "(280.0, 5)", "(1120.0, 5)",
                                   "(0.0, 6)", "(140.0, 6)", "(210.0, 6)", "(280.0, 6)", "(1120.0, 6)",
                                   "(0.0, 7)", "(140.0, 7)", "(210.0, 7)", "(280.0, 7)", "(1120.0, 7)",
                                   "(0.0, 8)", "(140.0, 8)", "(210.0, 8)", "(280.0, 8)", "(1120.0, 8)",
                                   "(0.0, 9)", "(140.0, 9)", "(210.0, 9)", "(280.0, 9)", "(1120.0, 9)",
                                   "(0.0, 10)", "(140.0, 10)", "(210.0, 10)", "(280.0, 10)", "(1120.0, 10)",
                                   "(0.0, 11)", "(140.0, 11)", "(210.0, 11)", "(280.0, 11)", "(1120.0, 11)",
                                   "(0.0, 12)", "(140.0, 12)", "(210.0, 12)", "(280.0, 12)", "(1120.0, 12)"]
    else:
        confusion_matrix_labels = cm_labels

    col_predicted = leaderboard["Column Predicted"].unique()[0]

    # Get predictions and true_labels of the run that we are interested in:
    runid = leaderboard.loc[(leaderboard["RL1s"] == rl1s) &
                            (leaderboard["Data and Split Description"] == percent_train_data),
                            "Run ID"].iloc[0]
    preds = pd.read_csv(os.path.join(th_results_path,
                                     "test_harness_results/runs/run_{}/testing_data.csv").format(runid))

    # we can use the labels parameter of confusion_matrix to to reorder or select a subset of labels.
    # If none is given, those that appear at least once in y_true or y_pred are used in sorted order.
    cm = pd.DataFrame(confusion_matrix(y_true=preds[col_predicted],
                                       y_pred=preds["{}_predictions".format(col_predicted)],
                                       labels=confusion_matrix_labels,
                                       normalize=normalize),
                      columns=confusion_matrix_labels,
                      index=confusion_matrix_labels)
    cm[col_predicted] = cm.index

    if col_predicted == "(conc, time)":
        cm['conc'], cm['time'] = cm["(conc, time)"].str.split(", ", 1).str
        cm['conc'] = [x.split("(", 1)[1] for x in cm['conc']]
        cm['time'] = [x.split(")", 1)[0] for x in cm['time']]
        cm.drop(columns=["(conc, time)"], inplace=True)
        cols_we_want_colors_for = ["conc", "time"]
    else:
        cols_we_want_colors_for = [col_predicted]

    class_cols = [c for c in cm.columns.values.tolist() if c not in cols_we_want_colors_for]

    color_cols = []
    for col in cols_we_want_colors_for:
        unique_meta_vals = cm[col].unique()
        lut = dict(zip(unique_meta_vals, sns.hls_palette(len(unique_meta_vals), l=0.5, s=0.8)))
        color_col = "color_{}".format(col)
        cm[color_col] = cm[col].map(lut)
        color_cols.append(color_col)
    cm.drop(columns=cols_we_want_colors_for, inplace=True)
    print(cm)
    print()

    if cbar:
        row_colors = cm[color_cols]
    else:
        row_colors = None
    cmap = sns.clustermap(cm[class_cols], row_colors=row_colors,
                          xticklabels=True, yticklabels=True, cmap="Greens")
    ax2 = cmap.ax_heatmap
    ax2.set_xlabel('Predicted labels')
    ax2.set_ylabel('True labels')
    cbar = ax2.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1, 0))
    if rl1s == "RL1s Used":
        ax2.set_title('Confusion Matrix: {} of Training Data Used, RL1s included in features. '
                      'Normalize = {}'.format(percent_train_data, normalize))
    elif rl1s == "RL1s Not Used":
        ax2.set_title('Confusion Matrix: {} of Training Data Used, RL1s Not included in features. '
                      'Normalize = {}'.format(percent_train_data, normalize))
    else:
        raise ValueError("rl1s must be equal to 'RL1s Used' or 'RL1s Not Used'")
    ax2.set_xticklabels(ax2.get_xmajorticklabels(), rotation=90, fontsize=7)
    ax2.set_yticklabels(ax2.get_ymajorticklabels(), rotation=0, fontsize=7)
    cmap.ax_col_dendrogram.set_visible(False)
    plt.show()


def wasserstein_heatmap(train_bank, percent_train_data=0.01):
    train_df, _ = train_test_split(train_bank, train_size=percent_train_data,
                                   random_state=5, stratify=train_bank[['(conc, time)']])
    relevant_df = train_df[["(conc, time)", "RL1-A"]].copy()
    c_t_tuples = list(relevant_df["(conc, time)"].unique())
    matrix = pd.DataFrame(columns=c_t_tuples, index=c_t_tuples)
    for c_t_1 in c_t_tuples:
        for c_t_2 in c_t_tuples:
            wd = wasserstein_distance(relevant_df.loc[relevant_df["(conc, time)"] == c_t_1, "RL1-A"],
                                      relevant_df.loc[relevant_df["(conc, time)"] == c_t_2, "RL1-A"])
            matrix[c_t_1][c_t_2] = float(wd)
    matrix = matrix.astype(float)
    print(matrix.shape)
    print()

    cmap = sns.clustermap(matrix, xticklabels=True, yticklabels=True, cmap="Greens")
    ax2 = cmap.ax_heatmap
    ax2.set_title('Wasserstein Matrix: {} of Training Data Used.'.format(percent_train_data))
    ax2.set_xticklabels(ax2.get_xmajorticklabels(), rotation=90, fontsize=7)
    ax2.set_yticklabels(ax2.get_ymajorticklabels(), rotation=0, fontsize=7)
    cmap.ax_col_dendrogram.set_visible(False)
    plt.show()


def main():
    # ------------------------------------------------------------------------------------------------------------------
    # initial groupings analysis:

    leaderboard = pd.read_html("test_harness_results/custom_classification_leaderboard.html")[0]
    leaderboard.loc[leaderboard['Num Features Used'] == 12, 'RL1s'] = 'RL1s Used'
    leaderboard.loc[leaderboard['Num Features Used'] == 9, 'RL1s'] = 'RL1s Not Used'

    # # create plot of percent train data vs. performance
    # ax1 = plt.subplot()
    # sns.scatterplot(leaderboard["Data and Split Description"], leaderboard["Balanced Accuracy"],
    #                 hue=leaderboard["RL1s"], ax=ax1)
    # ax1.set_xlabel("Percent of Training Data Used")
    # plt.ylim(0, 1)
    # plt.show()

    # # Create confusion matrices and clustermaps:
    # conf_matrix_and_clustermap(leaderboard, True, 0.01, 'true')
    # conf_matrix_and_clustermap(leaderboard, True, 0.40, 'true')
    # conf_matrix_and_clustermap(leaderboard, False, 0.01, 'true')
    # conf_matrix_and_clustermap(leaderboard, False, 0.40, 'true')

    # Wasserstein Distance
    # train_bank = pd.read_csv("train_bank.csv")
    # print("Shape of train_bank: {}".format(train_bank.shape))
    # print()
    # wasserstein_heatmap(train_bank, 0.01)
    # wasserstein_heatmap(train_bank, 0.40)

    # ------------------------------------------------------------------------------------------------------------------
    # ethanol groupings analysis:

    e_leaderboard = pd.read_html("ethanol_classes_results/test_harness_results/custom_classification_leaderboard.html")[0]
    e_leaderboard.loc[e_leaderboard['Num Features Used'] == 12, 'RL1s'] = 'RL1s Used'
    e_leaderboard.loc[e_leaderboard['Num Features Used'] == 9, 'RL1s'] = 'RL1s Not Used'
    print(e_leaderboard)
    print()

    # create plot of percent train data vs. performance
    ax1 = plt.subplot()
    sns.scatterplot(e_leaderboard["Data and Split Description"], e_leaderboard["Balanced Accuracy"],
                    hue=e_leaderboard["RL1s"], ax=ax1, s=55)
    ax1.set_xlabel("Percent of Training Data Used")
    plt.ylim(0, 1)
    plt.show()

    # Create ethanol-grouped confusion matrices and clustermaps:
    cm_labels = [0.0, 140.0, 210.0, 280.0, 1120.0]
    # conf_matrix_and_clustermap(e_leaderboard, True, 0.40, 'true', cm_labels,
    #                            os.path.join(os.getcwd(), "ethanol_classes_results"), cbar=False)

    # conf_matrix_and_clustermap(leaderboard, True, 0.01, 'true')


if __name__ == '__main__':
    main()
