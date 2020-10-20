import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras.optimizers import SGD
from collections import OrderedDict
from keras.models import Sequential
from keras.regularizers import l1_l2
from keras.layers import Dense, Dropout, Flatten
from sklearn.metrics import accuracy_score
from tensorflow.python.ops import gen_array_ops
from pysd2cat.analysis.live_dead_pipeline.names import Names as n
from pysd2cat.analysis.live_dead_pipeline.ld_pipeline_classes import LiveDeadPipeline

# gets rid of the annoying SettingWithCopyWarnings
# set this equal to "raise" if you feel like debugging SettingWithCopyWarnings.
pd.options.mode.chained_assignment = None  # default="warn"

col_idx = OrderedDict([(n.label, 0), ("inducer_concentration", 1), ("timepoint", 2), ("percent_live", 3)])

# tf.compat.v1.disable_eager_execution()
print(tf.__version__)
print("Eager Execution = {}\n".format(tf.executing_eagerly()))


def main():
    # Load, prepare, and generate labels for data
    ldp = LiveDeadPipeline(x_strain=n.yeast, x_treatment=n.ethanol, x_stain=1,
                           y_strain=None, y_treatment=None, y_stain=None)
    ldp.load_data()
    ldp.condition_method()
    labeled_df = ldp.labeled_data_dict[n.condition_method]
    df = pd.merge(ldp.x_df,
                  labeled_df[["arbitrary_index", "label_predictions"]].rename(columns={"label_predictions": n.label}),
                  on="arbitrary_index")

    df.rename(columns={"ethanol": "inducer_concentration", "time_point": "timepoint"}, inplace=True)
    df["timepoint"] = df["timepoint"] / 2

    features = n.morph_cols + n.sytox_cols

    df = df[features + ["inducer_concentration", "timepoint", "label"]]

    # add CFUs
    # using these lines as a temporary way to get the CFU data we want (CFUs = "ground truth")
    ldp_stain = LiveDeadPipeline(x_strain=n.yeast, x_treatment=n.ethanol, x_stain=1,
                                 y_strain=None, y_treatment=None, y_stain=None)
    ldp_stain.load_data()
    cfu_data = ldp_stain.cfu_df[["inducer_concentration", "timepoint", "percent_live"]]
    # need to mean the cfu percent_live column over "inducer_concentration" and "timepoint", due to multiple replicates
    cfu_means = cfu_data.groupby(by=["inducer_concentration", "timepoint"], as_index=False).mean()
    # print(cfu_means, "\n")
    df = pd.merge(df, cfu_means, how="inner", on=["inducer_concentration", "timepoint"])  # might want to change to left join

    print("\n" * 10, "*" * 200, "\n")
    print("df:\n", df, "\n")
    print(df["label"].value_counts(), "\n")

    features = n.morph_cols + n.sytox_cols
    X = df[features]
    Y = df[col_idx.keys()]
    print("X:\n", X, "\n")
    print("Y:\n", Y, "\n")

    # Begin keras model
    print("\n----------- Begin Keras Labeling Booster Model -----------\n")
    model = labeling_booster_model(input_shape=len(features))
    model.fit(X, Y, epochs=100, batch_size=1024)  # TODO: use generator instead of matrices
    class_predictions = (model.predict(X) > 0.5).astype("int32")
    training_accuracy = accuracy_score(y_true=Y[n.label], y_pred=class_predictions)
    print("\nTraining Accuracy = {}%\n".format(round(100 * training_accuracy, 2)))

    # TODO: add early stopping and a validation set, also use generator

    # TODO: add debug mode using following code
    '''
    # debugging joint_loss
    print("Debugging Loss Functions\n")
    debug_Y = Y.sample(n=2273, random_state=5)
    print(debug_Y, "\n")
    debug_label_conds_cfus = tf.convert_to_tensor(debug_Y)
    debug_y_pred = tf.convert_to_tensor(debug_Y[n.label].sample(frac=1))  # random labels for y_pred, used for testing functions
    loss = joint_loss(label_conds_cfus=debug_label_conds_cfus, y_pred=debug_y_pred)
    print(loss)
    # model.fit(tf.convert_to_tensor(X), debug_label_conds_cfus, epochs=10, batch_size=64)
    '''


def bin_cross(label_conds_cfus, y_pred):
    y_true = label_conds_cfus[:, col_idx[n.label]]
    y_true = tf.expand_dims(y_true, axis=1)  # necessary to get y_true in the same shape as y_pred
    return tf.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred)


def cfu_loss(label_conds_cfus, y_pred):
    cfu_percent_live = label_conds_cfus[:, col_idx["percent_live"]]
    cfu_percent_live = tf.expand_dims(cfu_percent_live, axis=1)  # necessary to get cfu_percent_live in the same shape as y_pred

    condition_indices = [col_idx["inducer_concentration"], col_idx["timepoint"]]
    conditions = tf.gather(label_conds_cfus, condition_indices, axis=1)

    y_pred = tf.sigmoid((y_pred - 0.5) * 1000)

    uniques, idx, count = gen_array_ops.unique_with_counts_v2(conditions, [0])

    num_unique = tf.size(count)
    sums = tf.math.unsorted_segment_sum(data=y_pred, segment_ids=idx, num_segments=num_unique)
    lengths = tf.cast(count, tf.float32)
    # print("lengths:\n", lengths, "\n")
    pred_percents = 100.0 * tf.divide(sums, lengths)  # check if I need to do multiplication via backend function
    # print("pred_percents:\n", pred_percents, "\n")

    pred_percents_mean = tf.math.reduce_mean(pred_percents)
    percents_live_mean = tf.math.reduce_mean(cfu_percent_live)
    diff = pred_percents_mean - percents_live_mean  # TODO: check if need backend function
    # print(pred_percents_mean, percents_live_mean, diff)

    return K.abs(diff)


def joint_loss(label_conds_cfus, y_pred):
    loss_bin_cross = bin_cross(label_conds_cfus=label_conds_cfus, y_pred=y_pred)
    loss_cfu = cfu_loss(label_conds_cfus=label_conds_cfus, y_pred=y_pred)

    return 0.99 * loss_bin_cross + 0.001 * loss_cfu
    # return loss_bin_cross


# can potentially branch forward scatter from side scatter and color.
# And then concat and then allow them to mix after they mix within their groups.
def labeling_booster_model(input_shape=None):
    model = Sequential()
    model.add(Dropout(0.1, input_shape=(input_shape,)))
    wr = l1_l2(l2=0.02, l1=0)
    model.add(Dense(units=32, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.3))
    model.add(Dense(units=16, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.1))
    model.add(Dense(units=8, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.1))
    model.add(Dense(units=4, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.1))
    wr = l1_l2(l2=0.02, l1=0)
    model.add(Dense(units=1, activation='sigmoid', kernel_regularizer=wr))
    model.add(Flatten())
    model.compile(loss=joint_loss, optimizer="Adam",
                  metrics=[joint_loss, bin_cross, cfu_loss], run_eagerly=True)
    # TODO: figure out how to add accuracy to metrics. Not trivial due to 4D nature of our Y.
    print(model.summary())
    print()
    return model


if __name__ == '__main__':
    main()
