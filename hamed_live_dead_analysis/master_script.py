import os
import sys
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from hamed_live_dead_analysis.pipeline_class import LiveDeadPipeline, ComparePipelines
from hamed_live_dead_analysis.names import Names as n

matplotlib.use("tkagg")
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)


def main():
    # ldp = LiveDeadPipeline(x_strain=n.yeast, x_treatment=n.ethanol, x_stain=0,
    #                        y_strain=None, y_treatment=None, y_stain=None)
    # ldp.load_data()
    # ldp.plot_distribution(channel=n.sytox_cols[0])
    # ldp.thresholding_method()
    # ldp.condition_method(live_conditions=None,
    #                      dead_conditions=[
    #                          {n.ethanol: 1120.0, n.time: n.timepoints[-1]},
    #                          {n.ethanol: 280.0, n.time: n.timepoints[-1]}
    #                      ]
    #                      )
    # ldp.cluster_method(n_clusters=2)
    # ldp.evaluate_performance(n.thresholding_method)
    # ldp.evaluate_performance(n.condition_method)
    # ldp.evaluate_performance(n.cluster_method)
    # ldp.plot_percent_live_over_conditions(n.condition_method)
    # ldp.plot_features_over_conditions(n.thresholding_method, kdeplot=True)

    ldp1 = LiveDeadPipeline(x_strain=n.yeast, x_treatment=n.ethanol, x_stain=0,
                            y_strain=n.yeast, y_treatment=n.ethanol, y_stain=0)
    ldp1.load_data()
    ldp1.thresholding_method()
    # ldp1.plot_features_over_conditions(n.thresholding_method, kdeplot=False, sample_fraction=0.01)
    # ldp1.plot_distribution(channel=n.sytox_cols[0], plot_x=True, plot_y=True, num_bins=50,
    #                         drop_zeros=True)

    ldp2 = LiveDeadPipeline(x_strain=n.yeast, x_treatment=n.ethanol, x_stain=1,
                            y_strain=n.yeast, y_treatment=n.ethanol, y_stain=1)
    ldp2.load_data()
    ldp2.thresholding_method()

    cp = ComparePipelines(ldp1, ldp2)
    cp.compare_plots_of_features_over_conditions(labeling_method_1=n.thresholding_method,
                                                 labeling_method_2=n.thresholding_method,
                                                 sample_fraction=1,
                                                 )


if __name__ == '__main__':
    main()
