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


def bin_cross(label_and_conds, y_pred):
    y_true = label_and_conds[n.label]
    return K.binary_crossentropy(y_true - y_pred)


def cfu_loss(label_and_conds, y_pred):
    cfu_data = pd.read_csv("/Users/he/PycharmProjects/SD2/pysd2cat/src/pysd2cat/analysis/"
                           "live_dead_pipeline/experiment_data/cfu_data/processed_and_combined_cfus.csv")
    cfu_data = cfu_data[["inducer_concentration", "timepoint", "percent_live"]]

    print(cfu_data)
    print()

    ratio_df = pd.DataFrame(columns=["ethanol", n.time, n.num_live, n.num_dead, n.percent_live])
    for tr in list(label_and_conds["ethanol"].unique()):
        for ti in list(label_and_conds[n.time].unique()):
            num_live = len(label_and_conds.loc[(label_and_conds["ethanol"] == tr) & (label_and_conds[n.time] == ti) & (
                    label_and_conds["label"] == 1)])
            num_dead = len(label_and_conds.loc[(label_and_conds["ethanol"] == tr) & (label_and_conds[n.time] == ti) & (
                    label_and_conds["label"] == 0)])
            ratio_df.loc[len(ratio_df)] = [tr, ti, num_live, num_dead, float(num_live) / (num_live + num_dead)]

    print(ratio_df)

    # use mean abs error like below
    # return K.mean(K.abs(y_true - y_pred))


def joint_loss(label_and_conds, y_pred):
    loss_1 = bin_cross(label_and_conds, y_pred)
    loss_2 = cfu_loss(label_and_conds, y_pred)
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
                  metrics=['accuracy'])
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
    features = n.morph_cols + n.sytox_cols
    X = df[features]
    Y = df[[n.label, n.ethanol, n.time]]

    cfu_loss(Y, Y[n.label])
    sys.exit(0)

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
