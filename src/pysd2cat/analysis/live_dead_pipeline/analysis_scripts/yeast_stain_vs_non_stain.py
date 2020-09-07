import os
import sys
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from pysd2cat.analysis.live_dead_pipeline.ld_pipeline_classes import LiveDeadPipeline
from pysd2cat.analysis.live_dead_pipeline.names import Names as n
from pysd2cat.analysis.live_dead_pipeline.experiment_data.cfu_data.process_and_combine_cfu_data import prep_2019_cfu_data

matplotlib.use("tkagg")
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', None)

strains = [n.yeast, n.ecoli, n.bacillus]


def overlaid_time_series_plot(concatenated_ratio_df, treatment="ethanol", style_col="was stain used?",
                              style_order=None, font_scale=1.0, title=None, tight=True, cfu_data=None):
    """
    Takes in a dataframe that has been labeled and generates a time-series plot of
    percent alive vs. time, colored by treatment amount.
    This serves as a qualitative metric that allows us to compare different methods of labeling live/dead.
    """
    concatenated_ratio_df[n.time] = concatenated_ratio_df[n.time] / 2.0

    matplotlib.use("tkagg")
    sns.set(style="ticks", font_scale=font_scale, rc={"lines.linewidth": 3.0})
    num_colors = len(concatenated_ratio_df[treatment].unique())
    palette = sns.color_palette("bright", num_colors)
    treatment_levels = list(concatenated_ratio_df[treatment].unique())
    treatment_levels.sort()
    palette = dict(zip(treatment_levels, palette))

    if cfu_data is not None:
        markers = ["o", "s"]
        for idx, cd in enumerate(cfu_data):
            sp = sns.scatterplot(data=cd, x="treatment_time", y="percent_live", s=250, marker=markers[idx],
                                 hue="treatment_concentration", legend="full", palette=palette, alpha=0.8, zorder=50)

    lp = sns.lineplot(x=concatenated_ratio_df[n.time], y=concatenated_ratio_df[n.percent_live],
                      hue=concatenated_ratio_df[treatment], style=concatenated_ratio_df[style_col],
                      style_order=style_order, palette=palette, legend="full", zorder=1)

    legend = plt.legend(bbox_to_anchor=(1.01, 0.9), loc=2, borderaxespad=0.,
                        handlelength=4, markerscale=1.9)
    legend.get_frame().set_edgecolor('black')
    plt.ylim(-0.05, 1.05)
    lp.set_xticks(np.linspace(0, 6, num=13))
    # lp.set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    if title is None:
        plt.title("Predicted Live over Time, with and without Stain Features. Yeast (S288C or CENPK)")
    else:
        plt.title(title)
    if tight:
        plt.tight_layout()

    new_labels = ["Ethanol Concentration of\nFlow Predictions", "0%", "10%", "15%", "20%", "80%",
                  "\nStain Used by Model", "True", "False",
                  "\n\nEthanol Concentration of\n2019 CFUs", "0%", "15%", "80%",
                  "\nEthanol Concentration of\n2020 CFUs", "0%", "20%", "80%"]
    for t, l in zip(legend.texts, new_labels): t.set_text(l)

    plt.xlabel("Exposure Time (Hours)")
    plt.ylabel("Percent Live")

    plt.show()
