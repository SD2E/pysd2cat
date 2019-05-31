from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
import pandas as pd
from pysd2cat.data import pipeline

from harness.test_harness_class import TestHarness
from harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification
from harness.utils.names import Names
import logging

import os

l = logging.getLogger(__file__)
l.setLevel(logging.DEBUG)


def build_model_pd(classifier_df,
                   data_df = None,
                   input_cols = ['FSC-A', 'SSC-A', 'BL1-A', 'RL1-A', 'FSC-H', 'SSC-H',
                                 'BL1-H', 'RL1-H', 'FSC-W', 'SSC-W', 'BL1-W', 'RL1-W'],
                   output_cols = ["class_label"],
                   output_location='harness_results',
                   description="yeast_live_dead_dataframe"
                                 ):
    # print("Length of full DF", len(df))
    #print(classifier_df)
    
    l.debug("Splitting Test Harness Data ...")
    train, test = train_test_split(classifier_df, stratify=classifier_df['class_label'],
                                   test_size=0.2, random_state=5)
    th = TestHarness(output_location=output_location)

    data_df = data_df.copy()
    data_df.loc[:, 'class_label'] = data_df.index

    l.debug("Running Test Harness ...")
    #rf_classification_model = random_forest_classification(n_estimators=500)
    th.run_custom(#test_harness_models=rf_classification_model,
                  function_that_returns_TH_model=random_forest_classification,
                  dict_of_function_parameters={},
                       training_data=train, 
                       testing_data=test,
                       data_and_split_description=description,
                       cols_to_predict=output_cols,
                       index_cols=input_cols+output_cols,
                       feature_cols_to_use=input_cols, 
                       normalize=True, 
                       feature_cols_to_normalize=input_cols,
#                    feature_extraction=Names.RFPIMP_PERMUTATION,
                            feature_extraction=False,
                       predict_untested_data=data_df)

    l.debug("Extracting Test Harness Predictions ...")
    leader_board = pd.read_html(os.path.join(output_location, 'test_harness_results/custom_classification_leaderboard.html'))[0]
    leader_board = leader_board.sort_values(by=['Date', 'Time'], ascending=True)
    l.debug("Selecting run: " + str(leader_board.iloc[-1, :]))
    run = leader_board.loc[:,'Run ID'].iloc[-1]
    #print(run)
    run_path = os.path.join(output_location, 'test_harness_results/runs/', "run_" + run)
    predictions_path = os.path.join(run_path, 'predicted_data.csv')

    predictions_df = pd.read_csv(predictions_path, index_col=None)
    return predictions_df

def build_model(dataframe):
    #print(c_df_norm[0:5,:])
    #df_norm = dataframe
    
    X = dataframe.drop(columns=['class_label'])
    y = dataframe['class_label'].astype(int)

    #binarizer = Binarizer(threshold=0.0).fit(y)
    #y = binarizer.transform(y)[:,0]

    #train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
    train_X, val_X, train_y, val_y = train_test_split(X, y, stratify=y,
                                   test_size=0.2, random_state=5)


    #scaler = Normalizer().fit(train_X)
    scaler = StandardScaler().fit(train_X)
    train_X_norm = scaler.transform(train_X)


    
    #print(c_df.columns[1:12])
    #val_X = pd.DataFrame(val_X, columns = dataframe.columns[1:13])
    #val_y = pd.DataFrame(val_y, columns = [dataframe.columns[0]])

    # Define model
    #logreg = LogisticRegression()
    rf_model = RandomForestClassifier(random_state=1, class_weight = 'balanced',
                                     n_estimators=361,  criterion='entropy', min_samples_leaf=13, n_jobs=-1)

    # Fit model
    #logreg.fit(train_X, train_y)
    rf_model.fit(train_X_norm, train_y)

    #test_scaler = Normalizer().fit(train_X)
    #test_scaler = StandardScaler().fit(train_X)
    val_X_norm = scaler.transform(val_X)
    val_p = pd.DataFrame(rf_model.predict(val_X_norm), columns=['class_label'])
    error = mean_absolute_error(val_y, val_p)
    
    val_X_norm = pd.DataFrame(val_X_norm, columns=dataframe.columns[1:])
    
    return (rf_model, error, val_X_norm, val_p, scaler)

def get_threshold_pr(df, threshold):
    df['live'] = df['RL1-A'].apply(lambda x: x < threshold)
    return df['live'].mean()

def compute_mean_live(model, 
                      scaler,
                      data_dir, 
                      threshold=2000, 
                      ods=[0.0003],
                      media=['SC Media'],
                      circuits=['XOR', 'XNOR', 'OR', 'NOR', 'NAND', 'AND'], 
                      inputs=['00', '01', '10', '11'],
                      fraction=0.06):
    mean_live = pd.DataFrame()
    for circuit in circuits:        
        for input in inputs:   
            for od in ods:
                for m in media:
                    c_df = pipeline.get_strain_dataframe_for_classifier(circuit, input, od=od, media=m, data_dir=data_dir)
                    c_df = c_df.sample(frac=0.1, replace=True)
                    c_df_norm = scaler.transform(c_df.drop(columns=['output']))
                    c_df_y = model.predict(c_df_norm)
                    pr_live = get_threshold_pr(c_df, threshold)
                    record = {'circuit' : circuit,
                              'input' : input,
                              'od' : od,
                              'media' : m,
                              'classifier' : c_df_y.mean(), 
                              'threshold' : pr_live}
                    mean_live = mean_live.append(record, ignore_index=True)
    return mean_live

def predict_live_dead(df, model, scaler):
    print("Predicting live...")
    df_norm = scaler.transform(df)
    predictions = model.predict(df_norm)
    #predictions = model.predict(df)
    #print(predictions)
    preds = pd.DataFrame(predictions, columns = ['class_label'])
    #df.loc[:,'class_label'] = preds['class_label'].astype(int)
    #df['class_label'] = predictions
    df.loc[:,'class_label'] = predictions
    #print(df)
    return df
    
