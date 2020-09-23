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
from pysd2cat.analysis.live_dead_pipeline.ld_pipeline_classes import LiveDeadPipeline, ComparePipelines

# gets rid of the annoying SettingWithCopyWarnings
# set this equal to "raise" if you feel like debugging SettingWithCopyWarnings.
pd.options.mode.chained_assignment = None  # default="warn"

col_idx = OrderedDict([(n.label, 0), ("inducer_concentration", 1), ("timepoint", 2)])


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
    print(df, "\n")

    features = n.morph_cols + n.sytox_cols
    X = df[features]
    Y = df[col_idx.keys()]

    # Begin keras model
    print("\n----------- Begin Keras Labeling Booster Model -----------\n")
    model = labeling_booster_model(input_shape=len(features))
    model.fit(X, Y, epochs=10, batch_size=64)

    # evaluate the keras model
    evaluation = model.evaluate(X, Y)
    print(evaluation)

    # class_predictions = np.ndarray.flatten(model.predict_classes(X_val))


def bin_cross(y_true, y_pred):
    return tf.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred)


def cfu_loss(conditions, y_pred, loop_or_group="group"):
    # create ratio_df from conditions and y_pred
    conditions[n.label_preds] = y_pred

    # "loop" and "group" do the same thing, but group is much more succinct (and probably faster)
    if loop_or_group == "loop":
        ratio_df = pd.DataFrame(columns=["inducer_concentration", "timepoint", "percent_live_predictions"])
        for tr in list(conditions["inducer_concentration"].unique()):
            for ti in list(conditions["timepoint"].unique()):
                num_live = len(conditions.loc[(conditions["inducer_concentration"] == tr) & (
                        conditions["timepoint"] == ti) & (conditions[n.label_preds] == 1)])
                num_dead = len(conditions.loc[(conditions["inducer_concentration"] == tr) & (
                        conditions["timepoint"] == ti) & (conditions[n.label_preds] == 0)])
                ratio_df.loc[len(ratio_df)] = [tr, ti, 100 * float(num_live) / (num_live + num_dead)]
    elif loop_or_group == "group":
        ratio_df = conditions.groupby(by=["inducer_concentration", "timepoint"],
                                      as_index=False).apply(lambda x: 100 * x[n.label_preds].sum() /
                                                                      len(x)).reset_index(name="percent_live_predictions")
    else:
        raise NotImplementedError()

    # using these lines as a temporary way to get the CFU data we want (CFUs = "ground truth")
    ldp_stain = LiveDeadPipeline(x_strain=n.yeast, x_treatment=n.ethanol, x_stain=1,
                                 y_strain=None, y_treatment=None, y_stain=None)
    ldp_stain.load_data()
    cfu_data = ldp_stain.cfu_df[["inducer_concentration", "timepoint", "percent_live"]]

    cond_cols = ["inducer_concentration", "timepoint"]
    merged_df = pd.merge(ratio_df, cfu_data, on=cond_cols)
    merged_df_grouped = merged_df.groupby(cond_cols, as_index=False).mean()
    merged_df_grouped["diff"] = merged_df_grouped["percent_live_predictions"] - merged_df_grouped[n.percent_live]
    # print(merged_df_grouped, "\n")
    return K.mean(K.abs(merged_df_grouped["diff"]))


def joint_loss(label_and_conds, y_pred):
    y_true = label_and_conds[:, col_idx[n.label]]
    condition_indices = [col_idx["inducer_concentration"], col_idx["timepoint"]]
    conditions = tf.gather(label_and_conds, condition_indices, axis=1)
    loss_1 = bin_cross(y_true=y_true, y_pred=y_pred)
    loss_2 = cfu_loss(conditions=conditions, y_pred=y_pred)
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
                  metrics=[bin_cross, cfu_loss, "accuracy"], run_eagerly=True)
    return model


if __name__ == '__main__':
    main()
