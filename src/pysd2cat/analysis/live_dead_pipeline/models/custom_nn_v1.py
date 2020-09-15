import os
import sys
import pandas as pd
import keras.backend as K
from keras.optimizers import SGD
from keras.models import Sequential
from keras.regularizers import l1_l2
from keras.layers import Dense, Dropout
from pysd2cat.analysis.live_dead_pipeline.names import Names as n
from pysd2cat.analysis.live_dead_pipeline.ld_pipeline_classes import LiveDeadPipeline, ComparePipelines
from harness.th_model_classes.class_keras_classification import KerasClassification


def bin_cross(y_true, y_pred):
    # y_true = label_and_conds[n.label]
    return K.binary_crossentropy(y_true - y_pred)


def cfu_loss(conditions, y_pred):
    # create ratio_df from conditions and y_pred
    conditions[n.label_preds] = y_pred
    ratio_df = pd.DataFrame(columns=["inducer_concentration", "timepoint", "percent_live_predictions"])
    for tr in list(conditions["inducer_concentration"].unique()):
        for ti in list(conditions["timepoint"].unique()):
            num_live = len(conditions.loc[(conditions["inducer_concentration"] == tr) & (
                    conditions["timepoint"] == ti) & (conditions[n.label_preds] == 1)])
            num_dead = len(conditions.loc[(conditions["inducer_concentration"] == tr) & (
                    conditions["timepoint"] == ti) & (conditions[n.label_preds] == 0)])
            ratio_df.loc[len(ratio_df)] = [tr, ti, float(num_live) / (num_live + num_dead)]

    # using these lines as a temporary way to get the CFU data we want (CFUs = "ground truth")
    ldp_stain = LiveDeadPipeline(x_strain=n.yeast, x_treatment=n.ethanol, x_stain=1,
                                 y_strain=None, y_treatment=None, y_stain=None)
    ldp_stain.load_data()
    cfu_data = ldp_stain.cfu_df[["inducer_concentration", "timepoint", "percent_live"]]

    cond_cols = ["inducer_concentration", "timepoint"]
    merged_df = pd.merge(ratio_df, cfu_data, on=cond_cols)
    merged_df_grouped = merged_df.groupby(cond_cols, as_index=False).mean()
    merged_df_grouped["diff"] = merged_df_grouped["percent_live_predictions"] - merged_df_grouped[n.percent_live]

    return K.mean(K.abs(merged_df_grouped["diff"]))


def joint_loss(label_and_conds, y_pred):
    y_true = label_and_conds[n.label]
    conditions = label_and_conds[["inducer_concentration", "timepoint"]]
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
    model.compile(loss=joint_loss, optimizer="Adam",
                  metrics=[bin_cross, cfu_loss, "accuracy"])
    return model


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
    X = df[features]
    Y = df[[n.label, "inducer_concentration", "timepoint"]]

    # debugging cfu_loss
    # loss = cfu_loss(Y, Y[n.label])

    # Begin keras model
    print("----------- Begin Keras Labeling Booster Model -----------")
    model = labeling_booster_model(input_shape=len(features))

    model.fit(X, Y, epochs=10, batch_size=1000)

    # evaluate the keras model
    _, accuracy = model.evaluate(X, Y)
    print('Accuracy: %.2f' % (accuracy * 100))

    # class_predictions = np.ndarray.flatten(model.predict_classes(X_val))


if __name__ == '__main__':
    main()
