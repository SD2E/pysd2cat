import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from harness.test_harness_class import TestHarness
import seaborn as sns

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)


def main():
    print()
    full_df_cleaned = pd.read_csv("full_live_dead_df_cleaned.csv")
    print("Shape of full_df_cleaned: {}".format(full_df_cleaned.shape))
    print()
    print(full_df_cleaned)

    relevant_df = full_df_cleaned[["arbitrary_index", "kill_volume", "time_point"]]

    print(relevant_df)

    import matplotlib.pyplot as plt
    g = sns.FacetGrid(relevant_df, col="kill_volume")
    g = g.map(plt.hist, "time_point")

    plt.savefig("mygraph.png")


if __name__ == '__main__':
    main()
