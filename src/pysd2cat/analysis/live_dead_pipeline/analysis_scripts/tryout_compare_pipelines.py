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
    ldp1 = LiveDeadPipeline(x_strain=n.yeast, x_treatment=n.ethanol, x_stain=0,
                            y_strain=n.yeast, y_treatment=n.ethanol, y_stain=0)
    ldp1.load_data()
    ldp1.thresholding_method()
    ldp1.plot_features_over_conditions(n.thresholding_method, kdeplot=False, sample_fraction=0.1)
    # ldp1.plot_distribution(channel=n.sytox_cols[0], plot_x=True, plot_y=True, num_bins=50,
    #                         drop_zeros=True)

    ldp2 = LiveDeadPipeline(x_strain=n.yeast, x_treatment=n.ethanol, x_stain=1,
                            y_strain=n.yeast, y_treatment=n.ethanol, y_stain=1)
    ldp2.load_data()
    ldp2.thresholding_method()

    cp = ComparePipelines(ldp1, ldp2)
    cp.compare_plots_of_features_over_conditions(labeling_method_1=n.thresholding_method,
                                                 labeling_method_2=n.thresholding_method,
                                                 sample_fraction=0.1)


if __name__ == '__main__':
    main()
