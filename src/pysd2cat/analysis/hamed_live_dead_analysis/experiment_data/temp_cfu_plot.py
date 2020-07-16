import os
import sys
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

matplotlib.use("tkagg")
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 100)

treatment_time_values = {"PFA": 15, "Ethanol": 1}
treatment_time_units = {"PFA": "minutes", "Ethanol": "hours"}
treatment_colors = {"PFA": px.colors.qualitative.Plotly[1],
                    "Ethanol": px.colors.qualitative.Plotly[2]}


def box_plot(data_frame, treatment="PFA"):
    time_value = treatment_time_values[treatment]
    time_unit = treatment_time_units[treatment]
    treatment_color = treatment_colors[treatment]
    treatment_df = data_frame.loc[data_frame["treatment time"] == time_value]
    fig = px.box(data_frame=treatment_df, x="treatment + concentration", y="percent_killed",
                 color="treatment", facet_col="Strain",
                 title="{} Experiments (timepoint = {} {})".format(treatment, time_value, time_unit),
                 color_discrete_sequence=[px.colors.qualitative.Plotly[0], treatment_color])
    fig.update_layout(title_x=0.5, font=dict(size=20))

    # # the following code block removes all the repeated x-axis labels and replaces them with a single label
    # for axis in fig.layout:
    #     if type(fig.layout[axis]) == go.layout.XAxis:
    #         fig.layout[axis].title.text = ''
    #     fig.update_layout(
    #         annotations=list(fig.layout.annotations) + [go.layout.Annotation(xref="paper", yref="paper",
    #                                                                          text="Treatment + Concentration",
    #                                                                          showarrow=False,
    #                                                                          x=0.5, y=-0.075, font=dict(size=25))])
    fig.write_html("{}_box_plot.html".format(treatment))


def main():
    yeast_cfus = pd.read_csv("Duke_live-dead_CFUs.csv")
    yeast_cfus.drop(columns=["Unnamed: 13", "Unnamed: 14"], inplace=True)
    yeast_cfus["treatment concentration"].fillna(0, inplace=True)
    yeast_cfus["treatment concentration"] = yeast_cfus["treatment concentration"].astype(int)
    yeast_cfus["treatment + concentration"] = yeast_cfus["treatment"] + "_" + yeast_cfus["treatment concentration"].astype(str)
    yeast_cfus["treatment + concentration"] = [x[0] for x in yeast_cfus["treatment + concentration"].str.split("_0")]
    yeast_cfus.rename(columns={"% Killed": "percent_killed"}, inplace=True)
    yeast_cfus["percent_killed"] = yeast_cfus["percent_killed"].str.strip("%").astype(float)
    # replace negative percent_killed values with 0
    yeast_cfus.percent_killed[yeast_cfus.percent_killed < 0] = 0
    print(yeast_cfus)
    print()

    # Create box plots:
    box_plot(data_frame=yeast_cfus, treatment="PFA")
    box_plot(data_frame=yeast_cfus, treatment="Ethanol")

    '''
    yeast_cfus_grouped = yeast_cfus.groupby(by=["Strain", "treatment", "treatment time", "treatment + concentration"],
                                            as_index=False).agg({"percent_killed": ["mean", "std"]})
    yeast_cfus_grouped.columns = yeast_cfus_grouped.columns.map('_'.join)
    yeast_cfus_grouped.columns = yeast_cfus_grouped.columns.str.rstrip("_")
    yeast_cfus_grouped.rename(columns={"percent_killed_mean": "mean_percent_killed"}, inplace=True)
    print(yeast_cfus_grouped)
    print()

    # Split DataFrame into PFA and Ethanol:
    grouped_cfus_pfa = yeast_cfus_grouped.loc[yeast_cfus_grouped["treatment time"] == 15]
    grouped_cfus_ethanol = yeast_cfus_grouped.loc[yeast_cfus_grouped["treatment time"] == 1]

    # Create bar plots:
    pfa_bar = px.bar(data_frame=grouped_cfus_pfa, x="treatment + concentration", y="mean_percent_killed",
                     color="treatment", facet_col="Strain", error_y="percent_killed_std",
                     title="PFA Experiments (timepoint = 15 minutes)", opacity=1,
                     color_discrete_sequence=[px.colors.qualitative.Plotly[0],
                                              px.colors.qualitative.Plotly[1]])
    pfa_bar.update_layout(title_x=0.5)
    pfa_bar.write_html("pfa_bar_plot.html")

    ethanol_bar = px.bar(data_frame=grouped_cfus_ethanol, x="treatment + concentration", y="mean_percent_killed",
                         color="treatment", facet_col="Strain", error_y="percent_killed_std",
                         title="Ethanol Experiments (timepoint = 1 hour)", opacity=1,
                         color_discrete_sequence=[px.colors.qualitative.Plotly[0],
                                                  px.colors.qualitative.Plotly[2]])
    ethanol_bar.update_layout(title_x=0.5)
    ethanol_bar.write_html("ethanol_bar_plot.html")
    '''


if __name__ == '__main__':
    main()
