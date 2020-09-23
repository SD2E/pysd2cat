import os
import sys
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras.optimizers import SGD
from collections import OrderedDict
from keras.models import Sequential
from keras.regularizers import l1_l2
from keras.layers import Dense, Dropout, Flatten
from pysd2cat.analysis.live_dead_pipeline.names import Names as n
from pysd2cat.analysis.live_dead_pipeline.ld_pipeline_classes import LiveDeadPipeline

# gets rid of the annoying SettingWithCopyWarnings
# set this equal to "raise" if you feel like debugging SettingWithCopyWarnings.
pd.options.mode.chained_assignment = None  # default="warn"

col_idx = OrderedDict([(n.label, 0), ("inducer_concentration", 1), ("timepoint", 2), ("percent_live", 3)])

# tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())


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
    # df["condition"] = df["inducer_concentration"].astype(str) + "_" + df["timepoint"].astype(str)
    # df.drop(columns=["inducer_concentration", "timepoint"], inplace=True)
    print(df, "\n")

    # add CFUs
    # using these lines as a temporary way to get the CFU data we want (CFUs = "ground truth")
    ldp_stain = LiveDeadPipeline(x_strain=n.yeast, x_treatment=n.ethanol, x_stain=1,
                                 y_strain=None, y_treatment=None, y_stain=None)
    ldp_stain.load_data()
    cfu_data = ldp_stain.cfu_df[["inducer_concentration", "timepoint", "percent_live"]]
    # need to mean the cfu percent_live column over "inducer_concentration" and "timepoint", due to multiple replicates
    cfu_means = cfu_data.groupby(by=["inducer_concentration", "timepoint"], as_index=False).mean()
    print(cfu_means, "\n")
    df = pd.merge(df, cfu_means, how="inner", on=["inducer_concentration", "timepoint"])  # might want to change to left join
    print(df, "\n")

    features = n.morph_cols + n.sytox_cols
    X = df[features]
    Y = df[col_idx.keys()]

    '''
    # debugging joint_loss
    print("Debugging Loss Functions\n")
    debug_Y = Y.sample(n=2273, random_state=5)
    print(debug_Y, "\n")
    debug_label_conds_cfus = tf.convert_to_tensor(debug_Y)
    debug_y_pred = tf.convert_to_tensor(debug_Y[n.label].sample(frac=1))  # random labels for y_pred, used for testing functions
    loss = joint_loss(label_conds_cfus=debug_label_conds_cfus, y_pred=debug_y_pred)
    print(loss)
    '''

    # Begin keras model
    print("\n----------- Begin Keras Labeling Booster Model -----------\n")
    model = labeling_booster_model(input_shape=len(features))
    model.fit(X, Y, epochs=1, batch_size=32)
    # model.fit(tf.convert_to_tensor(X), debug_label_conds_cfus, epochs=10, batch_size=64)

    # evaluate the keras model
    evaluation = model.evaluate(X, Y)
    print(evaluation)

    # class_predictions = np.ndarray.flatten(model.predict_classes(X_val))


def bin_cross(y_true, y_pred):
    return tf.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred)


def cfu_loss(conditions, cfu, y_pred):
    # y_pred = tf.round(y_pred)

    print("conditions tensor:\n", conditions, "\n")
    print("cfu percent_live tensor:\n", cfu, "\n")
    print("y_pred tensor:\n", y_pred, "\n\n\n")
    print()
    # tf.print(conditions)
    # tf.print(cfu)
    # tf.print(y_pred)
    # print(tf.keras.backend.eval(y_pred))

    # uniques, idx, count = tf.unique_with_counts(conditions)
    # print(uniques)
    # print(idx)
    # print(count)

    from tensorflow.python.ops import gen_array_ops
    uniques, idx, count = gen_array_ops.unique_with_counts_v2(conditions, [0])
    print(uniques)
    print(idx)
    print(count)
    # print()

    num_unique = tf.size(count)
    sums = tf.math.unsorted_segment_sum(data=y_pred, segment_ids=idx, num_segments=num_unique)
    # print("sums:\n", sums, "\n")
    # ones = tf.ones(y_pred.shape)
    # lengths = tf.math.unsorted_segment_sum(data=ones, segment_ids=idx, num_segments=num_unique)
    lengths = tf.cast(count, tf.float32)
    # print("lengths:\n", lengths, "\n")
    percents = 100.0 * tf.divide(sums, lengths)  # check if I need to do multiplicaiton via backend function
    # print("percents:\n", percents, "\n")

    ratios_mean = tf.math.reduce_mean(percents)
    cfus_mean = tf.math.reduce_mean(cfu)
    diff = ratios_mean - cfus_mean
    # print(ratios_mean, cfus_mean, diff)

    return K.mean(K.abs(diff))


def joint_loss(label_conds_cfus, y_pred):
    y_true = label_conds_cfus[:, col_idx[n.label]]
    cfus = label_conds_cfus[:, col_idx["percent_live"]]
    condition_indices = [col_idx["inducer_concentration"], col_idx["timepoint"]]
    conditions = tf.gather(label_conds_cfus, condition_indices, axis=1)
    y_pred = K.flatten(y_pred)  # replace this?
    loss_1 = bin_cross(y_true=y_true, y_pred=y_pred)
    loss_2 = cfu_loss(conditions=conditions, cfu=cfus, y_pred=y_pred)
    return 0.5 * loss_1 + 0.5 * loss_2


def labeling_booster_model(input_shape=None):
    model = Sequential()
    model.add(Dropout(0.1, input_shape=(input_shape,)))
    wr = l1_l2(l2=0.02, l1=0)
    model.add(Dense(units=12, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.1))
    model.add(Dense(units=5, activation="relu", kernel_regularizer=wr))
    model.add(Dropout(0.1))
    wr = l1_l2(l2=0.02, l1=0)
    model.add(Dense(units=1, activation='sigmoid', kernel_regularizer=wr))
    model.add(Flatten())
    model.compile(loss=joint_loss, optimizer="Adam",
                  metrics=[bin_cross, "accuracy"], run_eagerly=True)  # removed cfu_loss because it wasn't working in metrics
    return model


if __name__ == '__main__':
    main()
