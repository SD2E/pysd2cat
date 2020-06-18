import os
import sys
import random
import inspect
import itertools
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from hamed_live_dead_analysis.names import Names as n
from harness.test_harness_class import TestHarness
from harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification

matplotlib.use("tkagg")
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)

# how is this different from os.path.dirname(os.path.realpath(__file__))?
current_dir_path = os.getcwd()


# TODO: figure out where/when dataframes should be copied or not
# TODO: add in cross-strain and cross-treatment labeling if it makes sense.
class LiveDeadPipeline:
    def __init__(self, x_strain=n.yeast, x_treatment=n.ethanol, x_stain=1,
                 y_strain=None, y_treatment=None, y_stain=None, use_previously_trained_model=False):
        """
        Assuming we will always normalize via Standard Scalar and log10-transformed data, so those are not arguments.
        """
        # TODO: add NotImplementedError checks
        self.x_strain = x_strain
        self.x_treatment = x_treatment
        self.x_stain = x_stain
        self.y_strain = self.x_strain if y_strain is None else y_strain
        self.y_treatment = self.x_treatment if y_treatment is None else y_treatment
        self.y_stain = self.x_stain if y_stain is None else y_stain

        self.x_experiment_id = n.exp_dict[(self.x_strain, self.x_treatment)]
        self.x_data_path = os.path.join(current_dir_path, n.exp_data_dir, self.x_experiment_id)
        self.y_experiment_id = n.exp_dict[(self.y_strain, self.y_treatment)]
        self.y_data_path = os.path.join(current_dir_path, n.exp_data_dir, self.y_experiment_id)

        if (self.x_stain == 0) or (self.y_stain == 0):
            self.feature_cols = n.morph_cols
        else:
            self.feature_cols = n.morph_cols + n.sytox_cols

        self.harness_path = os.path.join(current_dir_path, n.harness_output_dir)
        self.runs_path = os.path.join(self.harness_path, "test_harness_results/runs")
        self.labeled_data = None
        self.method = None

        self.output_dir_name = "({}_{}_{})_({}_{}_{})".format(self.x_strain, self.x_treatment, self.x_stain,
                                                              self.y_strain, self.y_treatment, self.y_stain)
        self.output_path = os.path.join(current_dir_path, n.pipeline_output_dir, self.output_dir_name)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    # ----- Preprocessing -----

    # TODO: save out intermediate files (e.g. normalized train/test split csvs) and check if they exist already when running pipeline
    # save them with consistent names in the folder specified by self.data_path. Append argument info to file names?
    def load_data(self):
        """
        All data coming in should already be log10-transformed, so no need to transform it.
        :return:
        :rtype:
        """
        x_df = pd.read_csv(os.path.join(self.x_data_path, n.data_file_name))
        if (self.y_strain == self.x_strain) & (self.y_treatment == self.x_treatment) & (self.y_stain == self.x_stain):
            y_df = x_df  # TODO: need to make sure this is ok, might need to do x_df.copy() instead
        else:
            y_df = pd.read_csv(os.path.join(self.y_data_path, n.data_file_name))

        # filter dfs based on if user specified stained data or unstained data
        x_df = x_df.loc[x_df[n.stain] == self.x_stain]
        y_df = y_df.loc[y_df[n.stain] == self.y_stain]

        # we want to have the data normalized before model runs because of data visualizations? Or actually is that what we don't want?
        # if we do want normalized, then should be fine to normalize all the data together (before train/test split),
        # because our models will re-normalize after train/test splitting, which should have the same effect (double check this)
        # x_scaler = StandardScaler()
        # y_scaler = StandardScaler()
        # x_df[self.feature_cols] = x_scaler.fit_transform(x_df[self.feature_cols])
        # y_df[self.feature_cols] = y_scaler.fit_transform(y_df[self.feature_cols])
        # print(x_df.head())
        # print()
        # print(y_df.head())
        # print()
        self.x_df = x_df.copy()
        self.y_df = y_df.copy()

    # ----- Building Blocks -----

    def cluster(self):
        """
        Clusters the data according to the algorithm you choose.
        Hamed look and see if we already have code that does this.
        """
        pass

    def confusion_matrix(self):
        pass

    def find_threshold(self):
        pass

    def invoke_test_harness(self, train_df, test_df, pred_df):
        print("initializing TestHarness object with this output_location: {}\n".format(self.harness_path))
        th = TestHarness(output_location=self.harness_path)
        th.run_custom(function_that_returns_TH_model=random_forest_classification,
                      dict_of_function_parameters={},
                      training_data=train_df,
                      testing_data=test_df,
                      data_and_split_description="method: {}, x_strain: {}, x_treatment: {}, x_stain: {},"
                                                 " y_strain: {}, y_treatment: {}, y_stain: {}".format(inspect.stack()[1][3], self.x_strain,
                                                                                                      self.x_treatment, self.x_stain,
                                                                                                      self.y_strain, self.y_treatment,
                                                                                                      self.y_stain),
                      cols_to_predict=n.label,
                      feature_cols_to_use=self.feature_cols,
                      # TODO: figure out how to resolve discrepancies between x_treatment and y_treatment, since col names will be different
                      index_cols=[n.index, self.x_treatment, n.time, n.stain],
                      normalize=True,
                      feature_cols_to_normalize=self.feature_cols,
                      feature_extraction="eli5_permutation",
                      predict_untested_data=pred_df)
        return th.list_of_this_instance_run_ids[-1]

    # ----- Exploratory Methods -----
    def plot_distribution(self, channel=n.sytox_cols[0]):
        sns.distplot(self.x_df[channel], bins=50, color="lightgreen", norm_hist=False, kde=False)
        plt.title("Histogram of {}".format(channel))
        plt.show()

    def scatterplot(self):
        pass
        # palette = itertools.cycle(sns.color_palette())
        # ets = np.array([[0.0, 1, 1],
        #                 [210.0, 1, 1],
        #                 [1120.0, 1, 1]])
        # for i in ets:
        #     ethanol = i[0]
        #     timepoint = i[1]
        #     stain = i[2]
        #     df_sub = self.x_df[(self.x_df['ethanol'] == ethanol) &
        #                     (self.x_df['time_point'] == timepoint) &
        #                     (self.x_df['stain'] == stain)]
        #     if negative_outlier_cutoff is not None:
        #         df_sub = df_sub.loc[df_sub[channel] >= negative_outlier_cutoff]
        #     print(len(df_sub))
        #
        #     sns.distplot(df_sub[channel], bins=50, color=next(palette), norm_hist=False, kde=False,
        #                  label="Eth: {}, Time: {}, Stain: {}".format(ethanol, timepoint, stain))
        #
        # plt.legend()
        # if negative_outlier_cutoff is not None:
        #     plt.title("Distributions of the {} channel. Removed outliers below {}.".format(channel, negative_outlier_cutoff))
        # else:
        #     plt.title("Distributions of the {} channel.".format(channel))
        # plt.show()

    # ----- Filtering Debris Methods -----

    # ----- Labeling methods -----

    def condition_method(self, live_conditions=None, dead_conditions=None):
        """
        Define certain tuples of (treatment, time-point) as live or dead.
            E.g. (0-treatment, final timepoint) = live, (max-treatment, final timepoint) = dead
        Train supervised model on those definitions, and predict live/dead for all points.
        Final product is dataframe with original data and predicted labels, which is set to self.predicted_data


        :param live_conditions:
        :type live_conditions: list of dicts
        :param dead_conditions:
        :type dead_conditions: list of dicts
        """
        self.method = "condition_method"
        if live_conditions is None:
            live_conditions = [{self.x_treatment: n.treatments_dict[self.x_treatment][0], n.time: n.timepoints[-1]}]
        if dead_conditions is None:
            dead_conditions = [{self.x_treatment: n.treatments_dict[self.x_treatment][-1], n.time: n.timepoints[-1]}]
        print(live_conditions)
        print(dead_conditions)
        print()

        # Label points according to live_conditions and dead_conditions
        # first obtain indexes of the rows that correspond to live_conditions and dead_conditions
        live_indexes = []
        dead_indexes = []
        for lt in live_conditions:
            live_indexes += list(self.x_df.loc[(self.x_df[list(lt)] == pd.Series(lt)).all(axis=1), n.index])
        for dt in dead_conditions:
            dead_indexes += list(self.x_df.loc[(self.x_df[list(dt)] == pd.Series(dt)).all(axis=1), n.index])
        labeled_indexes = live_indexes + dead_indexes

        # TODO: check if this should call .copy() or not
        labeled_df = self.x_df[self.x_df[n.index].isin(labeled_indexes)]
        labeled_df.loc[labeled_df[n.index].isin(live_indexes), n.label] = 1
        labeled_df.loc[labeled_df[n.index].isin(dead_indexes), n.label] = 0
        print(labeled_df.head())
        # mainly doing this split so Test Harness can run without balking (it expects testing_data)
        train_df, test_df = train_test_split(labeled_df, train_size=0.95, random_state=5,
                                             stratify=labeled_df[[self.x_treatment, n.time, n.label]])

        # Invoke Test Harness
        run_id = self.invoke_test_harness(train_df=train_df, test_df=test_df, pred_df=self.y_df)
        self.labeled_data = pd.read_csv(os.path.join(self.runs_path, "run_{}/predicted_data.csv".format(run_id)))

    def thresholding_method(self, channel=n.sytox_cols[0]):
        """
        Currently uses arithmetic mean of channel (default RL1-A).
        Since channels are logged, arithmetic mean is equivalent to geometric mean of original channel values.
        Final product is dataframe with original data and predicted labels, which is set to self.predicted_data
        """
        self.method = "thresholding_method"
        channel_values = list(self.x_df[channel])
        threshold = np.array(channel_values).mean()

        labeled_df = self.x_df.copy()
        labeled_df.loc[labeled_df[channel] >= threshold, n.label_preds] = 1
        labeled_df.loc[labeled_df[channel] < threshold, n.label_preds] = 0
        print(labeled_df.head())
        self.labeled_data = labeled_df

    # ----- Performance Evaluation -----

    def time_series_plot(self, labeled_df):
        """
        Takes in a dataframe that has been labeled and generates a time-series plot of
        percent alive vs. time, colored by treatment amount.
        This serves as a qualitative metric that allows us to compare different methods of labeling live/dead.
        """
        matplotlib.use("tkagg")
        ratio_df = pd.DataFrame(columns=[self.y_treatment, n.time, n.num_live, n.num_dead, n.percent_live])
        for tr in list(labeled_df[self.y_treatment].unique()):
            for ti in list(labeled_df[n.time].unique()):
                num_live = len(labeled_df.loc[(labeled_df[self.y_treatment] == tr) & (labeled_df[n.time] == ti) & (
                        labeled_df[n.label_preds] == 1)])
                num_dead = len(labeled_df.loc[(labeled_df[self.y_treatment] == tr) & (labeled_df[n.time] == ti) & (
                        labeled_df[n.label_preds] == 0)])
                ratio_df.loc[len(ratio_df)] = [tr, ti, num_live, num_dead, float(num_live) / (num_live + num_dead)]
        palette = sns.color_palette("mako_r", 5)
        sns.lineplot(x=ratio_df[n.time], y=ratio_df[n.percent_live], hue=ratio_df[self.y_treatment], palette=palette)
        plt.title("Predicted Live over Time using {}".format(self.method))  # TODO make title more descriptive
        # plt.show()
        # TODO: add make_dir_if_does_not_exist
        plt.savefig(os.path.join(current_dir_path, n.pipeline_output_dir,
                                 "time_series_({}_{}_{})_({}_{}_{})_{}.png".format(self.x_strain, self.x_treatment, self.x_stain,
                                                                                   self.y_strain, self.y_treatment, self.y_stain,
                                                                                   self.method)))
        return ratio_df

    def quantitative_metrics(self, labeled_df):
        """
        Takes in a dataframe that has been labeled and runs a consistent supervised model on a train/test split of the now-labeled data.
        The model’s performance will be a quantitative way to see if our labels actually make sense.
         For example, if I give random labels to the data, then no supervised model will perform well on that data.
         But if the labels line up with some “ground truth”, then a supervised model should be able to perform better.
         Todo: look into how people evaluate semi-supervised models.
        """

    def evaluate_performance(self):
        """
        Calls qualitative and quantitative methods for performance evaluation
        """
        ratio_df = self.time_series_plot(self.labeled_data)
        # print(ratio_df)

        # self.quantitative_metrics(self.labeled_data)


class ComparePipelines:
    pass
