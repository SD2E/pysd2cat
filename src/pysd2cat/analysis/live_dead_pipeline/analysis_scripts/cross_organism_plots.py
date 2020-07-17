import os
import matplotlib
import pandas as pd
from pysd2cat.analysis.live_dead_pipeline.ld_pipeline_classes import LiveDeadPipeline
from pysd2cat.analysis.live_dead_pipeline.names import Names as n
from pysd2cat.analysis.live_dead_pipeline.analysis_scripts.yeast_stain_vs_non_stain import overlaid_time_series_plot

matplotlib.use("tkagg")
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', None)

strains = [n.yeast, n.ecoli, n.bacillus]


def cross_organism_time_series_plot(train_strain=n.yeast, stain=1):
    list_of_ratio_dfs = []
    for test_strain in strains:
        if test_strain != n.yeast:  # I added this because yeast results didn't transfer to microbes and plots looked terrible
            train_test_dir = "({}_ethanol_{})_({}_ethanol_{})".format(train_strain, stain, test_strain, stain)
            ratio_df = pd.read_csv(os.path.join("pipeline_outputs/{}/ratio_df.csv".format(train_test_dir)))
            ratio_df["test_strain"] = test_strain
            relevant_cols = ["test_strain", "ethanol", "time_point", "predicted %live"]
            list_of_ratio_dfs.append(ratio_df[relevant_cols])
    concatenated = pd.concat(list_of_ratio_dfs)
    print(concatenated)
    overlaid_time_series_plot(concatenated_ratio_df=concatenated,
                              treatment="ethanol",
                              style_col="test_strain")


def main():
    run_models = False

    if run_models:
        for train_strain in strains:
            for test_strain in strains:
                ldp = LiveDeadPipeline(x_strain=train_strain, x_treatment=n.ethanol, x_stain=0,
                                       y_strain=test_strain, y_treatment=n.ethanol, y_stain=0)
                ldp.load_data()
                print(ldp.feature_cols, "\n")
                ldp.condition_method(live_conditions=None,
                                     dead_conditions=[
                                         {n.ethanol: n.treatments_dict[n.ethanol][train_strain][-1], n.time: n.timepoints[-1]},
                                         {n.ethanol: n.treatments_dict[n.ethanol][train_strain][-2], n.time: n.timepoints[-1]}
                                     ])
                ldp.evaluate_performance(n.condition_method)

    cross_organism_time_series_plot(train_strain=n.ecoli, stain=1)


if __name__ == '__main__':
    main()
