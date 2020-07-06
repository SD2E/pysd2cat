import os
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import itertools

matplotlib.use('TkAgg')
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)

# TODO: remind myself what this code does
def plot_flow(df_tot, ets, channel='log_RL1-A', negative_outlier_cutoff=None):
    palette = itertools.cycle(sns.color_palette())

    for i in ets:
        ethanol = i[0]
        timepoint = i[1]
        stain = i[2]
        df_sub = df_tot[(df_tot['ethanol'] == ethanol) &
                        (df_tot['time_point'] == timepoint) &
                        (df_tot['stain'] == stain)]
        if negative_outlier_cutoff is not None:
            df_sub = df_sub.loc[df_sub[channel] >= negative_outlier_cutoff]
        print(len(df_sub))

        sns.distplot(df_sub[channel], bins=50, color=next(palette), norm_hist=False, kde=False,
                     label="Eth: {}, Time: {}, Stain: {}".format(ethanol, timepoint, stain))

    plt.legend()
    if negative_outlier_cutoff is not None:
        plt.title("Distributions of the {} channel. Removed outliers below {}.".format(channel, negative_outlier_cutoff))
    else:
        plt.title("Distributions of the {} channel.".format(channel))
    plt.show()

    # df_sub[channel].hist(bins=100)
    # plt.title(' Eth: ' + str(ethanol) + ', Time: ' + str(timepoint) + ', Channel: ' + channel)
    # plt.show()


def plot_flow_2(df_tot):
    palette = "pastel"
    g = sns.FacetGrid(df_tot, row="strain", col="time_point", hue='ethanol',
                      margin_titles=True, palette=palette)
    g.map(plt.hist, 'log_RL1-A', bins=100, log=True)
    g.add_legend()
    plt.savefig("plot_flow_2_{}.png".format(palette))


def main():
    organism = "yeast"

    if organism == "yeast":
        df = pd.read_csv("datasets/yeast_train_bank.csv")
    elif organism == "basc":
        df = pd.read_csv("datasets/basc_train_bank.csv")
    elif organism == "ecoli":
        df = pd.read_csv("datasets/ecoli_train_bank.csv")
    else:
        raise NotImplementedError()

    print(df)
    print()
    ets = np.array([[0.0, 1, 1],
                    [210.0, 1, 1],
                    [1120.0, 1, 1]])
    # plot_flow(df, ets=ets, channel="log_RL1-A", negative_outlier_cutoff=0.1)
    # plot_flow(df, ets=ets, channel="log_RL1-H", negative_outlier_cutoff=None)
    # plot_flow(df, ets=ets, channel="log_RL1-W", negative_outlier_cutoff=None)

    if organism == "yeast":
        df_2 = df.copy()
        df_2["strain"] = "yeast"
        print(df_2.shape)
        df_2 = df_2.loc[df_2["ethanol"].isin([0.0, 210.0, 1120.0])]
        df_2 = df_2.loc[df_2["stain"] == 1]
        print(df_2.shape)
        plot_flow_2(df_2)


# print()
# full_df_cleaned = pd.read_csv("full_live_dead_df_cleaned.csv")
# print("Shape of full_df_cleaned: {}".format(full_df_cleaned.shape))
# print()
# print(full_df_cleaned)
#
# relevant_df = full_df_cleaned[["arbitrary_index", "kill_volume", "time_point"]]
#
# print(relevant_df)
#
# import matplotlib.pyplot as plt
# g = sns.FacetGrid(relevant_df, col="kill_volume")
# g = g.map(plt.hist, "time_point")
#
# plt.savefig("mygraph.png")


if __name__ == '__main__':
    main()
