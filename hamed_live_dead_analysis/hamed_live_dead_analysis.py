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
from six import string_types

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)
CWD = os.getcwd()


def make_list_if_not_list(obj):
    if not isinstance(obj, list):
        return [obj]
    else:
        return obj


def is_list_of_strings(obj):
    if isinstance(obj, pd.DataFrame):
        return False
    elif obj and isinstance(obj, list):
        return all(isinstance(elem, string_types) for elem in obj)
    else:
        return False


def retrieve_preds_and_labels(th_results_path, rl1s=True, percent_train_data=0.4, append_cols=None, run_id=None):
    """

    :param th_results_path: path to test harness results folder (please include test_harness_results)
    :param rl1s: if RL1 features were used or not in the model
    :param percent_train_data: what percentage of the training data was used
    :param append_cols: columns from original test data to append to the preds_and_labels DataFrame
        (will use arbitrary_index column to merge).
    :type append_cols: None|str|list
    :param run_id: if you know the run_id, can pass it in here. This will override the use of rl1s and percent_train_data.
    :return: DataFrame containing the true labels and predictions of the model run indicated
    """

    leaderboard = pd.read_html(os.path.join(th_results_path, "custom_classification_leaderboard.html"))[0]

    if run_id is None:
        if rl1s:
            num_features_used = 12
        else:
            num_features_used = 9
        run_id = leaderboard.loc[(leaderboard["Num Features Used"] == num_features_used) &
                                 (leaderboard["Data and Split Description"] == percent_train_data),
                                 "Run ID"].iloc[0]

    # Get predictions and true labels of the run that we are interested in:
    preds_and_labels = pd.read_csv(os.path.join(th_results_path, "runs/run_{}/testing_data.csv").format(run_id))

    preds_and_labels["rl1s"] = rl1s
    preds_and_labels["percent_train_data"] = percent_train_data
    preds_and_labels["run_id"] = run_id

    if append_cols is not None:
        append_cols = make_list_if_not_list(append_cols)
        assert is_list_of_strings(append_cols), "append_cols must be a string or a list of strings"

        test_df = pd.read_csv(os.path.join(CWD, "test_df.csv"))
        relevant_cols = ["arbitrary_index"] + append_cols
        test_df_relevant = test_df[relevant_cols]
        preds_and_labels = pd.merge(preds_and_labels, test_df_relevant, on="arbitrary_index")
        if "(conc, time)" in append_cols:
            preds_and_labels['conc'], preds_and_labels['time'] = preds_and_labels["(conc, time)"].str.split(", ", 1).str
            preds_and_labels['conc'] = [x.split("(", 1)[1] for x in preds_and_labels['conc']]
            preds_and_labels['time'] = [x.split(")", 1)[0] for x in preds_and_labels['time']]

    return preds_and_labels


def make_confusion_matrix(preds_and_labels, target_col, normalize='true', label_order=None):
    if label_order is None:
        label_order = ["(0.0, 1)", "(140.0, 1)", "(210.0, 1)", "(280.0, 1)", "(1120.0, 1)",
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

    # We can use the labels parameter of confusion_matrix to to reorder or select a subset of labels.
    # confusion_matrix takes the labels that appear in y_true and y_pred, and matches them to the labels you pass in.
    # If None is given, the labels that appear at least once in y_true or y_pred are used in sorted order.
    # In this function we don't allow for labels=None.
    cm = pd.DataFrame(confusion_matrix(y_true=preds_and_labels[target_col],
                                       y_pred=preds_and_labels["{}_predictions".format(target_col)],
                                       labels=label_order,
                                       normalize=normalize),
                      columns=label_order,
                      index=label_order)
    cm[target_col] = cm.index
    if target_col == "(conc, time)":
        cm['conc'], cm['time'] = cm["(conc, time)"].str.split(", ", 1).str
        cm['conc'] = [x.split("(", 1)[1] for x in cm['conc']]
        cm['time'] = [x.split(")", 1)[0] for x in cm['time']]
        cm.drop(columns=["(conc, time)"], inplace=True)

    # create string of confusion matrix information
    rl1s = preds_and_labels["rl1s"].iloc[0]
    percent_train_data = preds_and_labels["percent_train_data"].iloc[0]

    if rl1s:
        cm_info = ('Confusion Matrix: {} of Training Data Used, RL1s included in features. '
                   'Normalize = {}'.format(percent_train_data, normalize))
    elif rl1s is False:
        cm_info = ('Confusion Matrix: {} of Training Data Used, RL1s Not included in features. '
                   'Normalize = {}'.format(percent_train_data, normalize))
    else:
        raise ValueError("rl1s must be equal to 'RL1s Used' or 'RL1s Not Used'")

    return cm, cm_info


def make_clustermap(cm, plot_title, color_by_cols=None, color_rows=False):
    color_by_cols = make_list_if_not_list(color_by_cols)
    assert is_list_of_strings(color_by_cols), "append_cols must be a string or a list of strings"

    if color_by_cols is not None:
        cm_cols = [c for c in cm.columns.values.tolist() if c not in color_by_cols]
        # create color mappings for columns we want to color by
        color_cols = []
        for col in color_by_cols:
            unique_meta_vals = cm[col].unique()
            lut = dict(zip(unique_meta_vals, sns.hls_palette(len(unique_meta_vals), l=0.5, s=0.8)))
            color_col = "color_{}".format(col)
            cm[color_col] = cm[col].map(lut)
            color_cols.append(color_col)
        cm.drop(columns=color_by_cols, inplace=True)
        row_colors = cm[color_cols]
    else:
        cm_cols = cm.columns.values.tolist()
        row_colors = None

    if color_rows is False:
        row_colors = None

    cmap = sns.clustermap(cm[cm_cols], row_colors=row_colors,
                          xticklabels=True, yticklabels=True, cmap="Greens")
    ax2 = cmap.ax_heatmap
    ax2.set_xlabel('Predicted labels')
    ax2.set_ylabel('True labels')
    cbar = ax2.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1, 0))
    ax2.set_title(plot_title)
    ax2.set_xticklabels(ax2.get_xmajorticklabels(), rotation=90, fontsize=7)
    ax2.set_yticklabels(ax2.get_ymajorticklabels(), rotation=0, fontsize=7)
    cmap.ax_col_dendrogram.set_visible(False)
    plt.show()


def histogram_of_confusion_regions(preds_and_labels, target_col, cm, top_n=3, extra_cols=None):
    extra_cols = make_list_if_not_list(extra_cols)
    assert is_list_of_strings(extra_cols), "append_cols must be a string or a list of strings"
    if extra_cols is not None:
        cm_cols = [c for c in cm.columns.values.tolist() if c not in extra_cols]
    else:
        cm_cols = cm.columns.values.tolist()

    stacked = cm[cm_cols].stack().reset_index()
    stacked.columns = ["True Labels", "Predicted Labels", "Ratio"]
    stacked.sort_values(by="Ratio", ascending=False, inplace=True)
    # print(stacked)

    for n in range(top_n):
        row = stacked.iloc[n]
        true_label = row["True Labels"]
        pred_label = row["Predicted Labels"]
        ratio = round(row["Ratio"], 3)
        print(true_label, pred_label, ratio)

        conc = preds_and_labels.loc[preds_and_labels[target_col] == true_label]
        # TODO: this might not make sense for confusion regions that are not along the diagonal:
        conc_correct = conc.loc[conc["{}_predictions".format(target_col)] == pred_label]
        conc_correct = conc_correct.astype({"time": int})
        conc_correct.sort_values(by="time", inplace=True)
        sns.countplot(x="time", data=conc_correct, color="lightgreen")
        plt.title("Histogram of time-point distribution within this confusion region:\n "
                  "True Label: {},   Predicted Label: {},   Ratio: {}".format(true_label, pred_label, ratio))
        plt.show()


def conc_and_time_vs_conc_matrix():
    pass


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

    # create plot of percent train data vs. performance
    # ax1 = plt.subplot()
    # sns.scatterplot(e_leaderboard["Data and Split Description"], e_leaderboard["Balanced Accuracy"],
    #                 hue=e_leaderboard["RL1s"], ax=ax1, s=55)
    # ax1.set_xlabel("Percent of Training Data Used")
    # plt.ylim(0, 1)
    # plt.show()

    # Create ethanol-grouped confusion matrices and clustermaps:
    # cm_labels = [0.0, 140.0, 210.0, 280.0, 1120.0]
    # conf_matrix_and_clustermap(e_leaderboard, True, 0.40, 'true', cm_labels,
    #                            os.path.join(os.getcwd(), "ethanol_classes_results"), cbar=False)

    # conf_matrix_and_clustermap(leaderboard, True, 0.01, 'true')

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    conc_time_results_path = os.path.join(CWD, "conc_time_results/test_harness_results")
    conc_results_path = os.path.join(CWD, "conc_results/test_harness_results")

    conc_time_results = retrieve_preds_and_labels(th_results_path=conc_time_results_path,
                                                  rl1s=True, percent_train_data=1.0)
    conc_results = retrieve_preds_and_labels(th_results_path=conc_results_path,
                                             rl1s=True, percent_train_data=1.0, append_cols="(conc, time)")

    # ------------------------------------------------------------------------------------------------------------------
    # conc_time analysis

    # create plot of percent train data vs. performance
    # leaderboard = pd.read_html("conc_time_results/test_harness_results/custom_classification_leaderboard.html")[0]
    # leaderboard.loc[leaderboard['Num Features Used'] == 12, 'RL1s'] = 'RL1s Used'
    # leaderboard.loc[leaderboard['Num Features Used'] == 9, 'RL1s'] = 'RL1s Not Used'
    # ax1 = plt.subplot()
    # sns.scatterplot(leaderboard["Data and Split Description"], leaderboard["Balanced Accuracy"],
    #                 hue=leaderboard["RL1s"], ax=ax1, s=55)
    # ax1.set_xlabel("Percent of Training Data Used")
    # plt.ylim(0, 1)
    # plt.show()

    # cm, cm_info = make_confusion_matrix(conc_time_results, "(conc, time)", "true", label_order=None)
    # make_clustermap(cm, cm_info, ["conc", "time"], True)

    # ------------------------------------------------------------------------------------------------------------------
    # conc analysis

    # create plot of percent train data vs. performance
    # e_leaderboard = pd.read_html("conc_results/test_harness_results/custom_classification_leaderboard.html")[0]
    # e_leaderboard.loc[e_leaderboard['Num Features Used'] == 12, 'RL1s'] = 'RL1s Used'
    # e_leaderboard.loc[e_leaderboard['Num Features Used'] == 9, 'RL1s'] = 'RL1s Not Used'
    # ax1 = plt.subplot()
    # sns.scatterplot(e_leaderboard["Data and Split Description"], e_leaderboard["Balanced Accuracy"],
    #                 hue=e_leaderboard["RL1s"], ax=ax1, s=55)
    # ax1.set_xlabel("Percent of Training Data Used")
    # plt.ylim(0, 1)
    # plt.show()

    label_order = [0.0, 140.0, 210.0, 280.0, 1120.0]
    cm, cm_info = make_confusion_matrix(conc_results, "kill_volume", "true", label_order)
    # make_clustermap(cm, cm_info, "kill_volume", False)
    histogram_of_confusion_regions(conc_results, "kill_volume", cm, 3, "kill_volume")


if __name__ == '__main__':
    main()
