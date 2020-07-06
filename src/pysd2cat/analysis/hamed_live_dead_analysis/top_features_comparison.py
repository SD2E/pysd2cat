import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pysd2cat.analysis.hamed_live_dead_analysis.supervised_analysis import retrieve_preds_and_labels

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)
CWD = os.getcwd()


def get_timeseries_scatter(df, xcol, ycol, stain, time_points, ethanols, color_col='ethanol_predictions', frac=0.1, kdeplot=False,
                           organism='yeast'):
    num_points = {'dead': {k: [] for k in ethanols}, "live": {k: [] for k in ethanols}}
    fig, ax = plt.subplots(ncols=len(time_points), nrows=len(ethanols), figsize=(4 * len(time_points), 4 * len(ethanols)), dpi=200)
    for i, row in enumerate(ax):
        ethanol = ethanols[i]
        for j, col in enumerate(row):
            time_point = time_points[j]
            time = df.loc[df.time_point == time_point].time_point.iloc[0]
            # just realized that this is wrong! It should be subsetting for time-point too not just ethanol concentration!
            if stain is None:
                plot_df = df.loc[(df.ethanol == ethanol)]
            else:
                plot_df = df.loc[(df.stain == stain) & (df.ethanol == ethanol)]
            low_df = plot_df.loc[plot_df[color_col] == ethanols[0]]
            mid_ethanol = ethanols[int(len(ethanols) / 2) + 1]
            mid_df = plot_df.loc[plot_df[color_col] == mid_ethanol]
            high_df = plot_df.loc[plot_df[color_col] == ethanols[-1]]
            num_points['dead'][ethanol].append(len(high_df))
            num_points['live'][ethanol].append(len(low_df))
            low_df = low_df.sample(frac=frac)
            mid_df = mid_df.sample(frac=frac)
            high_df = high_df.sample(frac=frac)
            try:
                if kdeplot:
                    sns.kdeplot(low_df[xcol], low_df[ycol], ax=col, alpha=0.5, cmap="Blues", shade=True, label="Live", shade_lowest=False,
                                dropna=True)
                else:
                    col.scatter(low_df[xcol], low_df[ycol], c="Blue", label="pred_{}".format(ethanols[0]),
                                s=100, alpha=0.4, marker='o', edgecolor='black', linewidth='0')
            except Exception as e:
                pass
            try:
                if kdeplot:
                    sns.kdeplot(mid_df[xcol], mid_df[ycol], ax=col, alpha=0.5, cmap="Oranges", shade=True, label="Dead",
                                shade_lowest=False,
                                dropna=True)
                else:
                    col.scatter(mid_df[xcol], mid_df[ycol], c="Green", label="pred_{}".format(mid_ethanol),
                                s=100, alpha=0.4, marker='o', edgecolor='black', linewidth='0')
                col.legend()
            except Exception as e:
                pass
            try:
                if kdeplot:
                    sns.kdeplot(high_df[xcol], high_df[ycol], ax=col, alpha=0.5, cmap="Oranges", shade=True, label="Dead",
                                shade_lowest=False,
                                dropna=True)
                else:
                    col.scatter(high_df[xcol], high_df[ycol], c="Red", label="pred_{}".format(ethanols[-1]),
                                s=100, alpha=0.4, marker='o', edgecolor='black', linewidth='0')
                col.legend()
            except Exception as e:
                pass

            col.set_xlabel("{}".format(xcol))
            col.set_ylabel("{}".format(ycol))

            if organism == "yeast":
                col.set_xlim(0, 7)
                col.set_ylim(0, 7)
            elif organism == "basc":
                col.set_xlim(0, 7)
                col.set_ylim(0, 7)
            elif organism == "ecoli":
                col.set_xlim(0, 7)
                col.set_ylim(0, 7)
            else:
                raise NotImplementedError()

            col.set_title("Ethanol (uL): " + str(ethanol) + "Time (h): " + str(time))
            # break
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.title(organism)
    # plt.savefig("top_features_comparison_{}_Hs.png".format(organism))
    return num_points


def main():
    organism = "ecoli"

    print("organism is: {}".format(organism))

    if organism == "yeast":
        df = retrieve_preds_and_labels(th_results_path="new_results/test_harness_results",
                                       test_df_path="datasets/yeast_normalized_test_df.csv",
                                       append_cols=["time_point"], run_id="weXr7qkbY38Or")
        df_test = pd.read_csv("datasets/yeast_test_df.csv")[["arbitrary_index", "log_RL1-H", "log_SSC-H"]]
        df = pd.merge(df, df_test, on="arbitrary_index")
        ethanols = [0.0, 140.0, 210.0, 280.0, 1120.0]
        frac = 0.05
    elif organism == "basc":
        df = retrieve_preds_and_labels(th_results_path="new_results/test_harness_results",
                                       test_df_path="datasets/basc_normalized_test_df.csv",
                                       append_cols=["time_point"], run_id="wb1r3yVbD1EVr")
        df_test = pd.read_csv("datasets/basc_test_df.csv")[["arbitrary_index", "log_RL1-H", "log_SSC-H"]]
        df = pd.merge(df, df_test, on="arbitrary_index")
        ethanols = [0.0, 5.0, 10.0, 15.0, 40.0]
        frac = 1
    elif organism == "ecoli":
        df = retrieve_preds_and_labels(th_results_path="new_results/test_harness_results",
                                       test_df_path="datasets/ecoli_normalized_test_df.csv",
                                       append_cols=["time_point"], run_id="8y5zDzAbDNjkg")
        df_test = pd.read_csv("datasets/ecoli_test_df.csv")[["arbitrary_index", "log_RL1-H", "log_SSC-H"]]
        df = pd.merge(df, df_test, on="arbitrary_index")
        ethanols = [0.0, 5.0, 10.0, 15.0, 40.0]
        frac = 1
    else:
        raise NotImplementedError()

    # print(df)
    time_points = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # time_points = [1, 2]

    # stain is None because the data is already all from stain=1 samples, and there is no stain col
    # get_timeseries_scatter(df, "log_SSC-H", "log_RL1-H", stain=None, time_points=time_points,
    #                        ethanols=ethanols, color_col="ethanol_predictions", organism=organism,
    #                        frac=frac)

    # df.to_csv("stained_{}_dataframe".format(organism), index=False)


if __name__ == '__main__':
    main()
