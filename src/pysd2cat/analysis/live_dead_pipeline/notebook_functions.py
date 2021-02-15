import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import keras.backend as K
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.regularizers import l1_l2
from sklearn.metrics import accuracy_score, r2_score
from collections import OrderedDict, Counter
from tensorflow.python.ops import gen_array_ops
from keras.layers import Dense, Dropout, Flatten
from names import Names as n
from scipy.stats import gaussian_kde
from descartes import PolygonPatch
import alphashape
from sklearn.cluster import KMeans
from pathlib import Path
from harness.test_harness_class import TestHarness
from harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification
from harness.th_model_instances.hamed_models.weighted_logistic import weighted_logistic_classifier

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', None)
matplotlib.use("tkagg")
warnings.filterwarnings('ignore')

col_idx = OrderedDict([(n.label, 0), ("inducer_concentration", 1), ("timepoint", 2), ("percent_live", 3)])


def training_progress(fitted_model, metrics, num_plots, loss_name, plot_val=None):
    '''
    A function for plotting the model history by showing the metric and loss throughout training
    '''
    plt.subplot(1, num_plots, 2)
    plt.plot(fitted_model.history['loss'], label='train loss')
    if plot_val is not None:
        plt.plot(fitted_model.history['val_loss'], label='val loss')
    plt.title('Loss ({})'.format(loss_name))
    #     plt.legend()

    for i, metric in enumerate(metrics):
        subplot_number = i + 3
        plt.subplot(1, num_plots, subplot_number)
        plt.plot(fitted_model.history[metric], label=str(metric))
        if plot_val is not None:
            plt.plot(fitted_model.history[f'val_{metric}'], label=f'val_{metric}')
        plt.title(str(metric))
    #         plt.legend()

    plt.show()


def plot_model_result(model, fitted_model, metrics, X, loss_name, plot_val=None):
    '''
    plot the foe_probabilities of a model
    plot the training history
    show training accuracy
    '''
    num_plots = len(metrics) + 2
    probas = model.predict_proba(X)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, num_plots, 1)
    plt.hist(probas)
    plt.xlabel('predict_proba')
    training_progress(fitted_model, metrics, num_plots, loss_name, plot_val)


def bin_cross(label_conds_cfus, y_pred):
    y_pred = K.flatten(y_pred)
    y_true = label_conds_cfus[:, col_idx[n.label]]
    final_loss = tf.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred)
    return final_loss


def cfu_loss(label_conds_cfus, y_pred):
    cfu_percent_live = label_conds_cfus[:, col_idx["percent_live"]]
    cfu_percent_live = cfu_percent_live / 100.0
    condition_indices = [col_idx["inducer_concentration"], col_idx["timepoint"]]
    conditions = tf.gather(label_conds_cfus, condition_indices, axis=1)
    y_pred = K.flatten(y_pred)
    y_pred = tf.sigmoid((y_pred - 0.5) * 100)
    uniques, idx, count = gen_array_ops.unique_with_counts_v2(conditions, [0])
    num_unique = tf.size(count)
    pred_percents = tf.math.unsorted_segment_mean(data=y_pred, segment_ids=idx, num_segments=num_unique)
    cfu_percents = tf.math.unsorted_segment_mean(data=cfu_percent_live, segment_ids=idx, num_segments=num_unique)
    diff = tf.math.subtract(pred_percents, cfu_percents)
    abs_diff = K.abs(diff)
    abs_diff_mean = tf.math.reduce_mean(abs_diff)
    return abs_diff_mean


def joint_loss_wrapper(cfu_loss_weight):
    def joint_loss(label_conds_cfus, y_pred):
        loss_bin_cross = bin_cross(label_conds_cfus=label_conds_cfus, y_pred=y_pred)
        loss_cfu = cfu_loss(label_conds_cfus=label_conds_cfus, y_pred=y_pred)
        return loss_bin_cross + cfu_loss_weight * loss_cfu

    return joint_loss


def booster_model_v2(input_shape=None, loss=joint_loss_wrapper(2), metrics=None, lr=0.1):
    if metrics is None:
        metrics = [loss]

    model = Sequential()
    model.add(Dense(units=32, activation="relu", input_shape=(input_shape,)))
    model.add(Dropout(0.1))
    model.add(Dense(units=16, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(units=8, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss=loss, optimizer=Adam(lr=lr),
                  metrics=metrics, run_eagerly=True)
    return model


def run_model(model_function, lr, loss, metrics, X, Y, epochs, batch_size, verbose, shuffle, plot_type="scatter"):
    # reset model each time
    tf.compat.v1.reset_default_graph
    tf.keras.backend.clear_session()

    # create model
    model = model_function(input_shape=len(X.columns), loss=loss, metrics=metrics, lr=lr)
    #     print(model.summary())

    # fit model and predict
    start_time = time.time()
    fitted_model = model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=shuffle)
    predict_proba = model.predict(X)
    class_predictions = np.ndarray.flatten(predict_proba > 0.5).astype("int32")

    # print metrics
    training_accuracy = accuracy_score(y_true=Y[n.label], y_pred=class_predictions)
    print("\nModel Boosting took {} seconds".format(time.time() - start_time))
    print("Training Accuracy = {}%".format(round(100 * training_accuracy, 2)))
    print(Counter(class_predictions))

    # plot probabilities and losses
    plot_model_result(model=model, fitted_model=fitted_model,
                      metrics=[x.__name__ for x in metrics],
                      X=X, loss_name=loss.__name__, plot_val=None)

    # print summary of predictions vs. cfus
    preds_and_labels = Y.copy()
    preds_and_labels[n.label] = preds_and_labels[n.label] * 100
    preds_and_labels["nn_preds"] = class_predictions * 100
    preds_and_labels.rename(columns={"percent_live": "cfu_percent_live"}, inplace=True)
    groupby_means = preds_and_labels.groupby([n.inducer_concentration, n.timepoint]).mean()
    # print(groupby_means)

    condition_results = groupby_means.reset_index()

    # plot percent live over conditions
    #     plot_percent_live_over_conditions(condition_results, plot_type)

    return condition_results, preds_and_labels


def plot_percent_live_over_conditions(condition_results, plot_type="scatter",
                                      color_by=n.inducer_concentration, fig_height=10,
                                      title=""):
    percent_live_df = condition_results

    plt.figure(figsize=(15, fig_height))
    sns.set(style="ticks", font_scale=2.0, rc={"lines.linewidth": 3.0})
    color_levels = list(percent_live_df[color_by].unique())
    color_levels.sort()
    num_colors = len(color_levels)
    palette = dict(zip(color_levels, sns.color_palette("bright", num_colors)))

    if plot_type == "line":
        #         labels = sns.lineplot(x=percent_live_df[n.timepoint], y=percent_live_df[n.label],
        #                       hue=percent_live_df[color_by],
        #                       palette=palette, legend=False, zorder=1, style=True, dashes=[(2,2)])

        preds = sns.lineplot(x=percent_live_df[n.timepoint], y=percent_live_df["nn_preds"],
                             hue=percent_live_df[color_by],
                             palette=palette, legend="full", zorder=1)

        cfus = sns.scatterplot(x=percent_live_df[n.timepoint], y=percent_live_df["cfu_percent_live"],
                               hue=percent_live_df[color_by],
                               palette=palette, legend=False, zorder=1, s=250, marker="o")

    elif plot_type == "scatter":
        labels = sns.scatterplot(x=percent_live_df[n.timepoint] - 0.14, y=percent_live_df[n.label],
                                 hue=percent_live_df[color_by],
                                 palette=palette, legend="full", zorder=1, s=250, marker="s")

        preds = sns.scatterplot(x=percent_live_df[n.timepoint], y=percent_live_df["nn_preds"],
                                hue=percent_live_df[color_by],
                                palette=palette, legend=False, zorder=1, s=250, marker="^")

        cfus = sns.scatterplot(x=percent_live_df[n.timepoint] + 0.14, y=percent_live_df["cfu_percent_live"],
                               hue=percent_live_df[color_by],
                               palette=palette, legend=False, zorder=1, s=250, marker="o")
        plt.title("Square = Label Percent Live\nTriangle = Predicted Percent Live\nCircle = CFU Percent Live")


    elif plot_type == "mixed":
        labels = sns.scatterplot(x=percent_live_df[n.timepoint] - 0.01, y=percent_live_df[n.label],
                                 hue=percent_live_df[color_by],
                                 palette=palette, legend=False, zorder=1, s=250, marker="s")

        preds = sns.lineplot(x=percent_live_df[n.timepoint], y=percent_live_df["nn_preds"],
                             #                                 hue=percent_live_df[color_by],
                             palette=palette, legend=False, zorder=1)

        cfus = sns.scatterplot(x=percent_live_df[n.timepoint] + 0.01, y=percent_live_df["cfu_percent_live"],
                               hue=percent_live_df[color_by],
                               palette=palette, legend=False, zorder=1, s=250, marker="o")
        plt.title(title, fontweight="bold")
        plt.xlabel("Exposure Time")
        plt.ylabel("Percent Live")

    # visual settings
    plt.xlim(0, 6.5)
    plt.ylim(-5, 105)

    #     legend = plt.legend(bbox_to_anchor=(1.01, 0.9), loc=2, borderaxespad=0.,
    #                         handlelength=4, markerscale=1.8)
    #     legend.get_frame().set_edgecolor('black')
    plt.show()


def plot_per_cond(condition_results):
    for conc in condition_results[n.inducer_concentration].unique():
        temp = condition_results.loc[condition_results[n.inducer_concentration] == conc]
        plot_percent_live_over_conditions(temp, plot_type="mixed", color_by=n.timepoint, fig_height=5,
                                          title="Ethanol Concentration = {}".format(conc))
    plt.show()


def plot_prob_changes(concat, y_max):
    zero_to_zero = concat.loc[concat["change_type"] == "Remained Dead", "label_probs"]
    one_to_one = concat.loc[concat["change_type"] == "Remained Live", "label_probs"]
    zero_to_one = concat.loc[concat["change_type"] == "Changed from Dead to Live", "label_probs"]
    one_to_zero = concat.loc[concat["change_type"] == "Changed from Live to Dead", "label_probs"]

    plt.figure(figsize=(10, 10))
    sns.distplot(zero_to_zero, bins=100, label="Remained Dead")
    sns.distplot(zero_to_one, bins=100, label="Changed from Dead to Live")
    sns.distplot(one_to_zero, bins=100, label="Changed from Live to Dead")
    sns.distplot(one_to_one, bins=100, label="Remained Live")

    plt.legend()
    plt.ylim(0, y_max)
    legend = plt.legend(bbox_to_anchor=(1.01, 0.9), loc=2, borderaxespad=0.,
                        handlelength=4, markerscale=1.8)
    legend.get_frame().set_edgecolor('black')
    plt.show()


def get_all_run_info(df, X, pl, append_df_cols=None):
    if append_df_cols is not None:
        X = pd.concat([X, df[append_df_cols]], axis=1)
    concat = pd.concat([X, pl], axis=1)
    concat["nn_preds"] = concat["nn_preds"] / 100
    concat[n.label] = concat[n.label] / 100
    concat["changed"] = ~(concat["label"] == concat["nn_preds"])
    concat["change_type"] = "this_should_be_gone"
    concat.loc[(concat["label"] == 0) & (concat["nn_preds"] == 0), "change_type"] = "Remained Dead"
    concat.loc[(concat["label"] == 1) & (concat["nn_preds"] == 1), "change_type"] = "Remained Live"
    concat.loc[(concat["label"] == 0) & (concat["nn_preds"] == 1), "change_type"] = "Changed from Dead to Live"
    concat.loc[(concat["label"] == 1) & (concat["nn_preds"] == 0), "change_type"] = "Changed from Live to Dead"
    concat['nn_percent_live'] = concat.groupby([n.inducer_concentration,
                                                n.timepoint])["nn_preds"].transform('mean') * 100
    concat["label_probs"] = df["label_probs"]

    return concat


def generate_rf_labels_from_conditions(data_df, features,
                                       live_conditions=None, dead_conditions=None):
    df = data_df.copy()
    df["arbitrary_index"] = df.index

    if live_conditions is None:
        live_conditions = [{n.inducer_concentration: 0.0, n.timepoint: 6.0}]
    if dead_conditions is None:
        dead_conditions = [{n.inducer_concentration: 20.0, n.timepoint: 6.0},
                           {n.inducer_concentration: 80.0, n.timepoint: 6.0}]
    print("Conditions designated as Live: {}".format(live_conditions))
    print("Conditions designated as Dead: {}".format(dead_conditions))
    print()

    # Label points according to live_conditions and dead_conditions
    # first obtain indexes of the rows that correspond to live_conditions and dead_conditions
    live_indexes = []
    dead_indexes = []
    for lc in live_conditions:
        live_indexes += list(df.loc[(df[list(lc)] == pd.Series(lc)).all(axis=1), n.index])
    for dc in dead_conditions:
        dead_indexes += list(df.loc[(df[list(dc)] == pd.Series(dc)).all(axis=1), n.index])
    labeled_indexes = live_indexes + dead_indexes

    # TODO: check if this should call .copy() or not
    labeled_df = df[df[n.index].isin(labeled_indexes)]
    labeled_df.loc[labeled_df[n.index].isin(live_indexes), n.label] = 1
    labeled_df.loc[labeled_df[n.index].isin(dead_indexes), n.label] = 0

    th = TestHarness(output_location=Path("__file__"))
    th.run_custom(function_that_returns_TH_model=random_forest_classification,
                  dict_of_function_parameters={},
                  training_data=labeled_df,
                  testing_data=labeled_df,
                  description="asdf",
                  target_cols=n.label,
                  feature_cols_to_use=features,
                  index_cols=["arbitrary_index"],
                  normalize=False,
                  feature_cols_to_normalize=features,
                  feature_extraction=False,
                  predict_untested_data=df)

    run_id = th.list_of_this_instance_run_ids[-1]
    labeled_df = pd.read_csv("__file__/test_harness_results/runs/run_{}/predicted_data.csv".format(run_id))

    df = pd.merge(df, labeled_df, on="arbitrary_index", how="left")
    df.rename(columns={"label_predictions": n.label, "label_prob_predictions": "label_probs"}, inplace=True)

    X = df[features]
    Y = df[col_idx.keys()]

    return df, X, Y


def get_point_density_values(x, y):
    xy = np.vstack([x, y])
    return gaussian_kde(xy)(xy)


def get_conc_df_with_kde_values(run_info: pd.DataFrame, conc, features, cc="RL1-H"):
    """

    :param run_info:
    :param conc:
    :param features:
    :param cc: color channel name
    :return:
    """
    if conc == "all":
        # Getting KDE values for all the data takes a longgggg time, so going to take a sample:
        conc_df = run_info.sample(frac=0.05, random_state=5)
    else:
        conc_df = run_info.loc[run_info[n.inducer_concentration] == conc]

    # create unlogged features
    for f in features:
        conc_df["log_{}".format(f)] = conc_df[f]
        conc_df[f] = conc_df[f].pow(10)

    # calculate point densities for (SSC-A, FSC-A) and (cc, FSC-A)
    conc_df["kde_S_F"] = get_point_density_values(x=conc_df["FSC-A"], y=conc_df["SSC-A"])
    conc_df["kde_R_F"] = get_point_density_values(x=conc_df[cc], y=conc_df["SSC-A"])
    conc_df["kde_log_S_F"] = get_point_density_values(x=conc_df["log_FSC-A"], y=conc_df["log_SSC-A"])
    conc_df["kde_log_R_F"] = get_point_density_values(x=conc_df["log_{}".format(cc)], y=conc_df["log_SSC-A"])

    return conc_df


def are_points_left_of_line(point_1_that_defines_line: tuple, point_2_that_defines_line: tuple,
                            x_values, y_values):
    point_1_that_defines_line = np.array(point_1_that_defines_line)
    point_2_that_defines_line = np.array(point_2_that_defines_line)
    points = np.array(pd.concat([x_values, y_values], axis=1))
    if point_1_that_defines_line[1] < point_2_that_defines_line[1]:
        are_points_left_of = np.cross(points - point_1_that_defines_line,
                                      points - point_2_that_defines_line) > 0
    elif point_1_that_defines_line[1] > point_2_that_defines_line[1]:
        are_points_left_of = np.cross(points - point_1_that_defines_line,
                                      points - point_2_that_defines_line) < 0
    else:
        raise ValueError("You have defined a horizontal line.\n"
                         "The y-component of 'point_1_that_defines_line' and 'point_2_that_defines_line' cannot be equal!")

    return are_points_left_of


def get_percent_of_autogater_live_points_left_of_soa_line(conc_df: pd.DataFrame, point_1_that_defines_line: tuple,
                                                          point_2_that_defines_line: tuple, cc="RL1-H"):
    autogater_live_preds = conc_df.loc[conc_df["nn_preds"] == 1]
    x_values = autogater_live_preds["log_{}".format(cc)]
    y_values = autogater_live_preds["log_FSC-A"]
    line_params = ([point_1_that_defines_line[0], point_2_that_defines_line[0]],
                   [point_1_that_defines_line[1], point_2_that_defines_line[1]])

    left_of_line = are_points_left_of_line(point_1_that_defines_line=point_1_that_defines_line,
                                           point_2_that_defines_line=point_2_that_defines_line,
                                           x_values=x_values,
                                           y_values=y_values)

    fig = plt.figure(figsize=(2.5, 2.5))
    plt.plot(line_params[0], line_params[1], c="cyan")
    plt.scatter(x_values, y_values, c=left_of_line, cmap="bwr", s=1)
    plt.xlim(0)
    plt.ylim(0)
    plt.show()
    percent_left_of_line = (float(sum(left_of_line)) / float(len(left_of_line))) * 100.0
    return round(percent_left_of_line, 2)


def kde_scatter(conc_df, cc="RL1-H", logged=False, fraction_of_points_based_on_kde=0.9,
                point_1_that_defines_line: tuple = (2.4, 4), point_2_that_defines_line: tuple = (3.05, 5.75),
                n_bins=None, cmap="Spectral_r", pred_col=None):
    print("Percent of AutoGater live predictions that are left of the SOA dashed line: {}".format(
        get_percent_of_autogater_live_points_left_of_soa_line(conc_df=conc_df, cc=cc,
                                                              point_1_that_defines_line=point_1_that_defines_line,
                                                              point_2_that_defines_line=point_2_that_defines_line)))

    dot_size = 1
    lines_color = "cyan"
    line_params = ([point_1_that_defines_line[0], point_2_that_defines_line[0]],
                   [point_1_that_defines_line[1], point_2_that_defines_line[1]])

    conc_df = conc_df.copy()

    fig = plt.figure(figsize=(20, 9))
    num_plot_cols = 2
    ax1 = plt.subplot(1, num_plot_cols, 1)
    ax3 = plt.subplot(1, num_plot_cols, num_plot_cols)

    if logged:
        ssc = "log_SSC-A"
        cc = "log_{}".format(cc)
        fsc = "log_FSC-A"
        kde_sf = "kde_log_S_F"
        kde_rf = "kde_log_R_F"
    else:
        ssc = "SSC-A"
        fsc = "FSC-A"
        kde_sf = "kde_S_F"
        kde_rf = "kde_R_F"

    # Sort the points by density, so that the densest points appear on top
    conc_df = conc_df.sort_values(by=kde_sf)

    # switch to plotting bins if n_bins is specified
    if n_bins is not None:
        conc_df[kde_sf] = pd.qcut(conc_df[kde_sf], n_bins, labels=False)
        conc_df[kde_rf] = pd.qcut(conc_df[kde_rf], n_bins, labels=False)

    # First plot:
    ax1.scatter(conc_df[ssc], conc_df[fsc], c=conc_df[kde_sf], s=dot_size, cmap=cmap)
    if logged:
        ax1.set_xlabel("log(SSC-A)", fontsize=40)
        ax1.set_ylabel("log(FCS-A)", fontsize=40)
    else:
        ax1.set_xlabel(ssc)
        ax1.set_ylabel(fsc)

    # Get subset of data based on fraction specified by fraction_of_points_based_on_kde
    # also don't include points with 0 values in the subset because they are trash and mess up kmeans
    subset_df = conc_df.loc[conc_df[fsc] != 0]
    subset_df = subset_df.loc[subset_df[ssc] != 0]
    subset_df = subset_df.tail(int(len(subset_df) * fraction_of_points_based_on_kde))  # tail works because we sorted by kde earlier

    # Overlay alpha_shape of the subset we got on top of first plot
    # first doing a KMeans with 2 clusters to get an alpha shape around each cluster
    subset_df["kmeans"] = KMeans(n_clusters=2, random_state=5).fit(subset_df[[ssc, fsc]]).labels_

    # always get the cluster with higher mean FSC:
    cluster_0_fsc_mean = subset_df.loc[subset_df["kmeans"] == 0, fsc].mean()
    cluster_1_fsc_mean = subset_df.loc[subset_df["kmeans"] == 1, fsc].mean()
    if cluster_0_fsc_mean > cluster_1_fsc_mean:
        good_cluster = 0
    else:
        good_cluster = 1

    cluster_boundary = alphashape.alphashape(list(zip(subset_df.loc[subset_df["kmeans"] == good_cluster, ssc],
                                                      subset_df.loc[subset_df["kmeans"] == good_cluster, fsc])), 0)
    ax1.add_patch(PolygonPatch(cluster_boundary, alpha=1.0, ec=lines_color, fc="none", lw=5))

    # Third plot:
    subset_no_debris = subset_df.loc[subset_df["kmeans"] == good_cluster]
    ax3.scatter(subset_no_debris[cc], subset_no_debris[fsc],
                c=get_point_density_values(x=subset_no_debris[cc], y=subset_no_debris[fsc]),
                s=dot_size, cmap=cmap)
    if logged:
        ax3.set_xlabel("log({})".format(cc.strip("log_")), fontsize=40)
        ax3.set_ylabel("log(FSC-A)", fontsize=40)
    else:
        ax3.set_xlabel(cc)
        ax3.set_ylabel(fsc)
    ax3.set_ylim(ax1.get_ylim())
    ax3.set_xlim(right=5)
    # Overlay boundary line (eyeballing it like they do in SOA):
    ax3.plot(line_params[0], line_params[1], c=lines_color)

    ax1.set_title("Selection of Cellular Events", fontweight="bold", fontsize=40)
    ax3.set_title("Manual Selection of Live Cells", fontweight="bold", fontsize=40)

    if pred_col is not None:
        fig = plt.figure(figsize=(20, 20.5))
        ax4 = plt.subplot(2, 2, 1)
        ax5 = plt.subplot(2, 2, 2)
        ax6 = plt.subplot(2, 2, 3)
        ax7 = plt.subplot(2, 2, 4)

        ax4.scatter(conc_df.loc[conc_df[pred_col] == 1, ssc],
                    conc_df.loc[conc_df[pred_col] == 1, fsc],
                    c=get_point_density_values(x=conc_df.loc[conc_df[pred_col] == 1, ssc],
                                               y=conc_df.loc[conc_df[pred_col] == 1, fsc]),
                    s=dot_size, cmap=cmap)
        ax5.scatter(subset_df.loc[subset_df[pred_col] == 1, cc],
                    subset_df.loc[subset_df[pred_col] == 1, fsc],
                    c=get_point_density_values(x=subset_df.loc[subset_df[pred_col] == 1, cc],
                                               y=subset_df.loc[subset_df[pred_col] == 1, fsc]),
                    s=dot_size, cmap=cmap)
        ax6.scatter(conc_df.loc[conc_df[pred_col] == 0, ssc],
                    conc_df.loc[conc_df[pred_col] == 0, fsc],
                    c=get_point_density_values(x=conc_df.loc[conc_df[pred_col] == 0, ssc],
                                               y=conc_df.loc[conc_df[pred_col] == 0, fsc]),
                    s=dot_size, cmap=cmap)
        ax7.scatter(subset_df.loc[subset_df[pred_col] == 0, cc],
                    subset_df.loc[subset_df[pred_col] == 0, fsc],
                    c=get_point_density_values(x=subset_df.loc[subset_df[pred_col] == 0, cc],
                                               y=subset_df.loc[subset_df[pred_col] == 0, fsc]),
                    s=dot_size, cmap=cmap)
        ax5.plot(line_params[0], line_params[1], c=lines_color, linestyle="dashed")
        ax7.plot(line_params[0], line_params[1], c=lines_color, linestyle="dashed")

        #             ax4.set_title("Predicted Healthy Cells", fontweight="bold", fontsize=30)
        #             ax5.set_title("Predicted Healthy Cells", fontweight="bold", fontsize=30)
        #             ax6.set_title("Predicted Dead, Dying, and Debris", fontweight="bold", fontsize=30)
        #             ax7.set_title("Predicted Dead, Dying, and Debris", fontweight="bold", fontsize=30)

        ax4.set_xlim(ax1.get_xlim())
        ax5.set_xlim(ax3.get_xlim())
        ax6.set_xlim(ax1.get_xlim())
        ax7.set_xlim(ax3.get_xlim())
        if logged:
            ax4.set_xlabel("log(SSC-A)", fontsize=40)
            ax5.set_xlabel("log({})".format(cc.strip("log_")), fontsize=40)
            ax6.set_xlabel("log(SSC-A)", fontsize=40)
            ax7.set_xlabel("log({})".format(cc.strip("log_")), fontsize=40)
        else:
            ax4.set_xlabel(ssc)
            ax5.set_xlabel(cc)
            ax6.set_xlabel(ssc)
            ax7.set_xlabel(cc)

        ax4.set_ylim(ax1.get_ylim())
        ax5.set_ylim(ax1.get_ylim())
        ax6.set_ylim(ax1.get_ylim())
        ax7.set_ylim(ax1.get_ylim())
        if logged:
            ax4.set_ylabel("log(FSC-A)", fontsize=40)
            ax5.set_ylabel("log(FSC-A)", fontsize=40)
            ax6.set_ylabel("log(FSC-A)", fontsize=40)
            ax7.set_ylabel("log(FSC-A)", fontsize=40)
        else:
            ax4.set_ylabel(fsc)
            ax5.set_ylabel(fsc)
            ax6.set_ylabel(fsc)
            ax7.set_ylabel(fsc)

    plt.subplots_adjust(hspace=0.4)
    plt.show()

    # Calculate SOA_preds by getting points left of boundary line in ax3
    cluster_of_interest = subset_df.loc[subset_df["kmeans"] == good_cluster]  # this gives us only the points in ax3
    cluster_of_interest["SOA_preds"] = are_points_left_of_line(point_1_that_defines_line=point_1_that_defines_line,
                                                               point_2_that_defines_line=point_2_that_defines_line,
                                                               x_values=cluster_of_interest[cc],
                                                               y_values=cluster_of_interest[fsc]).astype(int)
    soa_live_indices = cluster_of_interest.loc[cluster_of_interest["SOA_preds"] == 1].index
    conc_df["SOA_preds"] = 0
    conc_df.loc[soa_live_indices, 'SOA_preds'] = 1

    return conc_df


def summary_table_of_results(kde_df):
    summary_table = kde_df.groupby([n.inducer_concentration, n.timepoint], as_index=False).mean()

    name_weakly = "Weakly Supervised Model (RF)"
    name_cfu = "CFUs"
    name_weakly_boosted = "AutoGater"
    name_SOA = "State of the Art"

    summary_table.rename(columns={"label": name_weakly,
                                  "cfu_percent_live": name_cfu,
                                  "nn_preds": name_weakly_boosted,
                                  "SOA_preds": name_SOA}, inplace=True)

    summary_table = summary_table[[n.inducer_concentration, n.timepoint,
                                   name_weakly, name_cfu, name_weakly_boosted, name_SOA]]
    # summary_table = summary_table.loc[summary_table[n.timepoint].isin([0.5, 3.0, 6.0])]
    summary_table[name_weakly] = summary_table[name_weakly] * 100
    summary_table[name_weakly_boosted] = summary_table[name_weakly_boosted] * 100
    summary_table[name_SOA] = summary_table[name_SOA] * 100

    # Calculating R^2 between CFUs and models
    fig = plt.figure(figsize=[5, 5])
    sns.scatterplot(summary_table["CFUs"], summary_table["State of the Art"], legend="full", label="SOA")
    sns.scatterplot(summary_table["CFUs"], summary_table["AutoGater"], legend="full", label="AutoGater")
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    plt.ylabel("Model")
    plt.show()
    soa_r2 = r2_score(summary_table["State of the Art"], summary_table["CFUs"])
    autogater_r2 = r2_score(summary_table["AutoGater"], summary_table["CFUs"])
    print("R-Squared between CFUs and State of the Art: {}".format(soa_r2))
    print("R-Squared between CFUs and AutoGater: {}".format(autogater_r2))
    print()

    summary_table[name_weakly] = summary_table[name_weakly].astype(int).astype(str) + "%"
    summary_table[name_cfu] = summary_table[name_cfu].astype(int).astype(str) + "%"
    summary_table[name_weakly_boosted] = summary_table[name_weakly_boosted].astype(int).astype(str) + "%"
    summary_table[name_SOA] = summary_table[name_SOA].astype(int).astype(str) + "%"

    summary_table[n.inducer_concentration] = summary_table[n.inducer_concentration].astype(str)
    summary_table[n.timepoint] = summary_table[n.timepoint].astype(str)

    # summary_table_styled = summary_table.style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
    # summary_table_styled.set_properties(**{'text-align': 'center'}).hide_index()

    return summary_table


def percent_live_comparison_plot(summary_table):
    # Getting summary_table into the right format
    concat_summary_table = pd.concat([summary_table[["inducer_concentration",
                                                     "timepoint", "CFUs"]].assign(model='CFUs'),
                                      summary_table[["inducer_concentration",
                                                     "timepoint", "AutoGater"]].assign(model='AutoGater'),
                                      summary_table[["inducer_concentration",
                                                     "timepoint", "State of the Art"]].assign(model='State of the Art')])
    concat_summary_table.rename(columns={"CFUs": "percent"}, inplace=True)

    concat_summary_table['percent'].update(concat_summary_table.pop('AutoGater'))
    concat_summary_table['percent'].update(concat_summary_table.pop('State of the Art'))
    concat_summary_table["timepoint"] = concat_summary_table["timepoint"].astype(float)
    concat_summary_table["percent"] = concat_summary_table["percent"].str.rstrip('%').astype('float')

    # Plotting code
    sns.set(style="ticks", font_scale=1.8, rc={"lines.linewidth": 3.0})
    for c in [str(x) for x in list(concat_summary_table[n.inducer_concentration].unique())]:
        plt.figure(figsize=(7, 5))

        sc = sns.scatterplot(data=concat_summary_table.loc[concat_summary_table["inducer_concentration"] == c],
                             x="timepoint", y="percent", s=500, alpha=.7, style="model",
                             hue="model", hue_order=["CFUs", "State of the Art", "AutoGater"], legend="full")

        legend = plt.legend(bbox_to_anchor=(1.01, 0.9), loc=2, borderaxespad=0.,
                            handlelength=4, markerscale=1.8)
        legend.get_frame().set_edgecolor('black')
        new_labels = ['CFU Derivation', 'Sytox Stain Method Prediction', 'AutoGater Prediction']
        for t, l in zip(legend.texts, new_labels): t.set_text(l)

        sc.set(xlim=(-0.5, 6.5), xlabel="Time Point (Hours)",
               ylim=(-5, 105), ylabel="Percent Live",
               xticks=[0, 0.5, 3, 6], xticklabels=[0, 0.5, 3, 6])
        sc.set_title("Ethanol Concentration of {}%".format(c), fontweight="bold")
        plt.show()