import os
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import keras.backend as K
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.models import Sequential
from keras.regularizers import l1_l2
from sklearn.metrics import accuracy_score
from collections import OrderedDict, Counter
from tensorflow.python.ops import gen_array_ops
from keras.layers import Dense, Dropout, Flatten
from pysd2cat.analysis.live_dead_pipeline.names import Names as n
from pysd2cat.analysis.live_dead_pipeline.models.cfu_neural_network.custom_nn_v1 import labeling_booster_model, \
    joint_loss, bin_cross, cfu_loss, cfu_loss_2

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', None)
# gets rid of the annoying SettingWithCopyWarnings
# set this equal to "raise" if you feel like debugging SettingWithCopyWarnings.
pd.options.mode.chained_assignment = None  # default="warn"

col_idx = OrderedDict([(n.label, 0), ("inducer_concentration", 1), ("timepoint", 2), ("percent_live", 3)])

# tf.compat.v1.disable_eager_execution()
print("Keras version: {}".format(keras.__version__))
print("Tensorflow version: {}".format(tf.__version__))
print("Eager Execution = {}\n".format(tf.executing_eagerly()))


def main():
    df = pd.read_csv("df_for_testing.csv")
    print("df:\n", df, "\n")
    print(df["label"].value_counts(), "\n")

    df = df.loc[df["timepoint"].isin([1.0])]
    df = df.loc[df["inducer_concentration"].isin([5.0])]
    print(df["label"].value_counts(), "\n")

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
    start_time = time.time()
    loss_to_use = bin_cross  # set loss function here
    model = labeling_booster_model(input_shape=len(features), loss=loss_to_use)

    dummy_x = X[:100]
    dummy_y = Y[:100]
    dummy_loss = model.train_on_batch(dummy_x, dummy_y)
    print(dummy_loss)

    # weight_grads = get_weight_grad(model, dummy_x, dummy_y)
    output_grad = get_layer_output_grad(model, dummy_x, dummy_y)

    # print(weight_grads)
    print(output_grad)


    """
    fitted_model = model.fit(X, Y, epochs=25, batch_size=128, verbose=True, shuffle=True, validation_split=0.1)
    predict_proba = model.predict(X)
    class_predictions = np.ndarray.flatten(predict_proba > 0.5).astype("int32")
    training_accuracy = accuracy_score(y_true=Y[n.label], y_pred=class_predictions)
    print("\nModel Boosting took {} seconds".format(time.time() - start_time))
    print("\nTraining Accuracy = {}%\n".format(round(100 * training_accuracy, 2)))
    print(Counter(class_predictions), "\n")

    plot_model_result(model=model, fitted_model=fitted_model,
                      metrics=[x.__name__ for x in [bin_cross, cfu_loss, joint_loss]],
                      train=df, test=df, feature_cols=features)
    """

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


def get_weight_grad(model, inputs, outputs):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad


def get_layer_output_grad(model, inputs, outputs, layer=-1):
    """ Gets gradient a layer output for given inputs and outputs"""
    grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad


def training_progress(fitted_model, metrics, num_plots):
    '''
    A function for plotting the model history by showing the metric and loss throughout training
    '''
    plt.subplot(1, num_plots, 2)
    plt.plot(fitted_model.history['loss'], label='train loss')
    plt.plot(fitted_model.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.legend()

    for i, metric in enumerate(metrics):
        subplot_number = i + 3
        print(subplot_number)
        plt.subplot(1, num_plots, subplot_number)
        plt.plot(fitted_model.history[metric], label=str(metric))
        plt.plot(fitted_model.history[f'val_{metric}'], label=f'val_{metric}')
        plt.title(str(metric))
        plt.legend()

    plt.show()


def plot_model_result(model, fitted_model, metrics, train, test, feature_cols):
    '''
    plot the foe_probabilities of a model
    plot the training history
    show training and testing accuracy
    '''
    num_plots = len(metrics) + 2
    probas = model.predict_proba(train[feature_cols])
    plt.figure(figsize=(15, 5))
    plt.subplot(1, num_plots, 1)
    plt.hist(probas)
    plt.xlabel('predict_proba')
    training_progress(fitted_model, metrics, num_plots)


if __name__ == '__main__':
    main()
