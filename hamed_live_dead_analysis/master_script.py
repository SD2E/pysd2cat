import os
import sys
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, MeanShift
from harness.test_harness_class import TestHarness
from harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification
from hamed_live_dead_analysis.names import Names
from hamed_live_dead_analysis.pipeline_class import LiveDeadPipeline

matplotlib.use("tkagg")
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)


def main():
    ldp = LiveDeadPipeline()
    ldp.load_data()
    ldp.tuple_method()
    ldp.evaluate_performance()


if __name__ == '__main__':
    main()
