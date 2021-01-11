import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras.optimizers import SGD
from collections import OrderedDict, Counter
from keras.models import Sequential
from keras.regularizers import l1_l2
from keras.layers import Dense, Dropout, Flatten
from sklearn.metrics import accuracy_score
from tensorflow.python.ops import gen_array_ops
from pysd2cat.analysis.live_dead_pipeline.names import Names as n

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', None)
# gets rid of the annoying SettingWithCopyWarnings
# set this equal to "raise" if you feel like debugging SettingWithCopyWarnings.
pd.options.mode.chained_assignment = None  # default="warn"

col_idx = OrderedDict([(n.label, 0), ("inducer_concentration", 1), ("timepoint", 2), ("percent_live", 3)])

# tf.compat.v1.disable_eager_execution()
print(tf.__version__)
print("Eager Execution = {}\n".format(tf.executing_eagerly()))


def main():
    df = pd.read_csv("df_for_testing.csv")
    print("df:\n", df, "\n")
    print(df["label"].value_counts(), "\n")

    # df = df.loc[df["timepoint"].isin([0.5, 3.0])]
    # df = df.loc[df["inducer_concentration"].isin([0.0, 80.0])]

    features = n.morph_cols
    X = df[features]
    Y = df[col_idx.keys()]
    print("X:\n", X, "\n")
    print("Y:\n", Y, "\n")
    # print(Y["inducer_concentration"].value_counts(), "\n")
    # print(Y["timepoint"].value_counts(), "\n")
    # print(Y["percent_live"].value_counts(), "\n")

    # Begin keras model
    print("\n----------- Begin Keras Labeling Booster Model -----------\n")
    model = labeling_booster_model(input_shape=len(features))
    model.fit(X, Y, epochs=100, batch_size=1024)  # TODO: use generator instead of matrices
    class_predictions = np.ndarray.flatten(model.predict(X) > 0.5).astype("int32")
    training_accuracy = accuracy_score(y_true=Y[n.label], y_pred=class_predictions)
    print("\nTraining Accuracy = {}%\n".format(round(100 * training_accuracy, 2)))
    print(Counter(class_predictions))

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
    print(label_conds_cfus)
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

    return K.abs(diff / 100.0)


def cfu_loss_2(label_conds_cfus, y_pred):
    cfu_percent_live = label_conds_cfus[:, col_idx["percent_live"]] / 100.0
    # cfu_percent_live = tf.expand_dims(cfu_percent_live, axis=1)

    diff = cfu_percent_live - K.flatten(y_pred)
    return K.abs(K.mean(diff))


def cfu_loss_1d(cfu_true, y_pred):
    pass


def joint_loss(label_conds_cfus, y_pred):
    loss_bin_cross = bin_cross(label_conds_cfus=label_conds_cfus, y_pred=y_pred)
    loss_cfu = cfu_loss(label_conds_cfus=label_conds_cfus, y_pred=y_pred)
    # loss_cfu = cfu_loss_2(label_conds_cfus=label_conds_cfus, y_pred=y_pred)

    return loss_bin_cross + 3 * loss_cfu
    # return loss_bin_cross
    # return loss_cfu


# can potentially branch forward scatter from side scatter and color.
# And then concat and then allow them to mix after they mix within their groups.
def labeling_booster_model(input_shape=None, loss=joint_loss):
    model = Sequential()
    model.add(Dropout(0.1, input_shape=(input_shape,)))
    # wr = l1_l2(l2=0.02, l1=0)
    wr = None
    model.add(Dense(units=32, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.3))
    model.add(Dense(units=16, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.1))
    model.add(Dense(units=8, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.1))
    model.add(Dense(units=4, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.1))
    # wr = l1_l2(l2=0.02, l1=0)
    model.add(Dense(units=1, activation='sigmoid', kernel_regularizer=wr))
    model.add(Flatten())
    model.compile(loss=loss, optimizer="Adam",
                  metrics=[joint_loss, bin_cross, cfu_loss, cfu_loss_2], run_eagerly=True)
    # TODO: figure out how to add accuracy to metrics. Not trivial due to 4D nature of our Y.
    print(model.summary())
    print()
    return model


def labeling_booster_model_1d(input_shape=None, loss=joint_loss):
    model = Sequential()
    model.add(Dropout(0.1, input_shape=(input_shape,)))
    # wr = l1_l2(l2=0.02, l1=0)
    wr = None
    model.add(Dense(units=32, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.3))
    model.add(Dense(units=16, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.1))
    model.add(Dense(units=8, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.1))
    model.add(Dense(units=4, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.1))
    # wr = l1_l2(l2=0.02, l1=0)
    model.add(Dense(units=1, activation='sigmoid', kernel_regularizer=wr))
    model.add(Flatten())
    model.compile(loss=cfu_loss_1d, optimizer="Adam",
                  metrics=[cfu_loss_1d], run_eagerly=True)
    # TODO: figure out how to add accuracy to metrics. Not trivial due to 4D nature of our Y.
    print(model.summary())
    print()
    return model


if __name__ == '__main__':
    main()
