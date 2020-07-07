import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)
matplotlib.use("tkagg")

# ---------------------------------- cluster-driven labeling ----------------------------------
# current_path = os.getcwd()
# th_results_path = os.path.join(current_path, "clustering_ld_definition_results/test_harness_results")
# leaderboard = pd.read_html(os.path.join(th_results_path, "custom_classification_leaderboard.html"))[0]
#
# run_id = leaderboard["Run ID"].iloc[0]
#
# # print(run_id)
#
# # Get predictions and true labels of the run that we are interested in:
# preds_and_labels = pd.read_csv(os.path.join(th_results_path, "runs/run_{}/testing_data.csv").format(run_id))
#
# labeled_df = pd.read_csv("labeled_df.csv")
# labeled_df.drop(columns=["label"], inplace=True)
#
# merged = pd.merge(preds_and_labels, labeled_df, on="arbitrary_index")
#
# print(merged)
#
# ratio_df = pd.DataFrame(columns=["ethanol", "time_point", "num_live", "num_dead", "proportion"])
#
# for e in list(merged["ethanol"].unique()):
#     for t in list(merged["time_point"].unique()):
#         num_live = len(merged.loc[(merged["ethanol"] == e) & (merged["time_point"] == t) & (merged["label_predictions"] == 1)])
#         num_dead = len(merged.loc[(merged["ethanol"] == e) & (merged["time_point"] == t) & (merged["label_predictions"] == 0)])
#         # print(num_live)
#         ratio_df.loc[len(ratio_df)] = [e, t, num_live, num_dead, float(num_live) / (num_live + num_dead)]
#
# print(ratio_df)
#
# palette = sns.color_palette("mako_r", 5)
# sns.lineplot(x=ratio_df["time_point"], y=ratio_df["proportion"], hue=ratio_df["ethanol"], palette=palette)
# plt.title("Predicted Live over Time")
# plt.show()

# ---------------------------------- treatment-driven labeling ----------------------------------
current_path = os.getcwd()
th_results_path = os.path.join(current_path, "ld_definition_results/test_harness_results")
leaderboard = pd.read_html(os.path.join(th_results_path, "custom_classification_leaderboard.html"))[0]
run_id = leaderboard["Run ID"].iloc[0]

# print(run_id)

# Get predictions and true labels of the run that we are interested in:
preds = pd.read_csv(os.path.join(th_results_path, "runs/run_{}/predicted_data.csv").format(run_id))

print(preds)

ratio_df = pd.DataFrame(columns=["ethanol", "time_point", "num_live", "num_dead", "proportion"])
for e in list(preds["ethanol"].unique()):
    for t in list(preds["time_point"].unique()):
        num_live = len(preds.loc[(preds["ethanol"] == e) & (preds["time_point"] == t) & (preds["label_predictions"] == 1)])
        num_dead = len(preds.loc[(preds["ethanol"] == e) & (preds["time_point"] == t) & (preds["label_predictions"] == 0)])
        # print(num_live)
        ratio_df.loc[len(ratio_df)] = [e, t, num_live, num_dead, float(num_live) / (num_live + num_dead)]

print(ratio_df)

palette = sns.color_palette("mako_r", 5)
sns.lineplot(x=ratio_df["time_point"], y=ratio_df["proportion"], hue=ratio_df["ethanol"], palette=palette)
plt.title("Predicted Live over Time (Treatment-Driven Method)")
plt.show()
