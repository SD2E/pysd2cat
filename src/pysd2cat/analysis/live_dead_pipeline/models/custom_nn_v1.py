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
    y_true = label_and_conds[n.label]
    conds = label_and_conds[[n.ethanol, n.time]]
    cfu_data = pd.read_csv("/Users/he/PycharmProjects/SD2/pysd2cat/src/pysd2cat/analysis/"
                           "live_dead_pipeline/experiment_data/cfu_data/sample_CFU_table.csv")
    cfu_data = cfu_data[["treatment_concentration", "treatment_time", "percent_live"]]
    cfu_data["treatment_time"] = cfu_data["treatment_time"] * 2

    print(cfu_data)
    return K.mean(K.abs(y_true - y_pred))


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
    model.compile(loss='binary_crossentropy', optimizer="Adam",
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
