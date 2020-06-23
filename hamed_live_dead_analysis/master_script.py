import os
import sys
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from hamed_live_dead_analysis.pipeline_class import LiveDeadPipeline
from hamed_live_dead_analysis.names import Names as n

matplotlib.use("tkagg")
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)


def main():
    ldp = LiveDeadPipeline(x_strain=n.yeast, x_treatment=n.ethanol, x_stain=1,
                           y_strain=None, y_treatment=None, y_stain=1)
    ldp.load_data()
    # ldp.plot_distribution()
    # ldp.condition_method()
    ldp.thresholding_method()
    ldp.evaluate_performance()


if __name__ == '__main__':
    main()
