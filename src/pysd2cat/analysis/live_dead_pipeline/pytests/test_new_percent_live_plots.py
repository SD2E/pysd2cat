import os
import matplotlib
import pandas as pd
from pathlib import Path
from pysd2cat.analysis.live_dead_pipeline.names import Names as n
from pysd2cat.analysis.live_dead_pipeline.ld_pipeline_classes import LiveDeadPipeline, ComparePipelines

matplotlib.use("tkagg")
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', None)


# # in 2019 data, add noise to the 15 and 80 ethanol concentrations so they don't overlap
# cfu_data_2019.loc[cfu_data_2019["treatment_concentration"] == 15, "percent_live"] += 1
# cfu_data_2019.loc[cfu_data_2019["treatment_concentration"] == 80, "percent_live"] -= 1

def main():
    strain = n.yeast

    # stain model
    ldp_stain = LiveDeadPipeline(x_strain=strain, x_treatment=n.ethanol, x_stain=1,
                                 y_strain=None, y_treatment=None, y_stain=None)
    ldp_stain.load_data()
    ldp_stain.condition_method(live_conditions=None,
                               dead_conditions=[
                                   {n.ethanol: 80.0, n.time: n.time_points[-1]},
                                   # {n.ethanol: 20.0, n.time: n.time_points[-1]},
                                   # {n.ethanol: 15.0, n.time: n.time_points[-1]}
                               ])

    # non-stain model
    ldp_no_stain = LiveDeadPipeline(x_strain=strain, x_treatment=n.ethanol, x_stain=0,
                                    y_strain=None, y_treatment=None, y_stain=None)
    ldp_no_stain.load_data()
    ldp_no_stain.condition_method(live_conditions=None,
                                  dead_conditions=[
                                      {n.ethanol: 80.0, n.time: n.time_points[-1]},
                                      # {n.ethanol: 20.0, n.time: n.time_points[-1]},
                                      # {n.ethanol: 15.0, n.time: n.time_points[-1]}
                                  ])
    print("\n\n\n\n\n\n\n")
    ldp_no_stain.plot_percent_live_over_conditions(n.condition_method)

    print("\n\n\n\n\n\n\n")
    cp = ComparePipelines(ldp_stain, ldp_no_stain)
    cp.compare_percent_live_plots(n.condition_method, n.condition_method)

    import sys
    sys.exit(0)

    # don't need multiple CFU files because we can just combine them into one and then have a column determine the marker shape

    # pivoted = concatenated.pivot_table(index=["ethanol", "time_point"], columns="was stain used?", values="predicted %live")
    # pivoted.reset_index(inplace=True)
    # pivoted.rename(columns={True: "True", False: "False"}, inplace=True)
    # print(pivoted)
    # print()
    # print(pivoted.columns)
    # print()
    #
    # sns.set(style="ticks", font_scale=2.0)
    # lm = sns.lmplot(data=pivoted, x="True", y="False", hue="ethanol", ci=None, legend=True)
    # lm = (lm.set_axis_labels("Predicted % Live When Stain is Used",
    #                          "Predicted % Live Without Use of Stain")
    #       .set(xlim=(-0.1, 1), ylim=(-0.1, 1)))
    #
    # plt.title("asdf\n")
    # plt.show()
    #
    # for e in pivoted["ethanol"].unique():
    #     group = pivoted.loc[pivoted["ethanol"] == e]
    #     x = group["True"]
    #     y = group["False"]
    #     pearson = pearsonr(x, y)[0]
    #     print(e, pearson)
    #     # sns.scatterplot(x=x, y=y)
    #     # plt.show()
    # print("overall pearson score: {}".format(pearsonr(pivoted["True"], pivoted["False"])))


if __name__ == '__main__':
    main()
