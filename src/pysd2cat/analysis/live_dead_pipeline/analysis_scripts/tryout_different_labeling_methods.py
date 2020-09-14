import os
import sys
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from pysd2cat.analysis.live_dead_pipeline.ld_pipeline_classes import LiveDeadPipeline, ComparePipelines
from pysd2cat.analysis.live_dead_pipeline.names import Names as n

matplotlib.use("tkagg")
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', None)


def main():
    ldp = LiveDeadPipeline(x_strain=n.yeast, x_treatment=n.ethanol, x_stain=0,
                           y_strain=None, y_treatment=None, y_stain=None)
    ldp.load_data()
    ldp.plot_distribution(channel=n.sytox_cols[0])
    ldp.thresholding_method()
    ldp.condition_method(live_conditions=None,
                         dead_conditions=[
                             {n.ethanol: 1120.0, n.time: n.time_points[-1]},
                             {n.ethanol: 280.0, n.time: n.time_points[-1]}
                         ]
                         )
    ldp.cluster_method(n_clusters=2)
    ldp.evaluate_performance(n.thresholding_method)
    ldp.evaluate_performance(n.condition_method)
    ldp.evaluate_performance(n.cluster_method)
    ldp.plot_percent_live_over_conditions(n.condition_method)
    ldp.plot_features_over_conditions(n.thresholding_method, kdeplot=True)


if __name__ == '__main__':
    main()
