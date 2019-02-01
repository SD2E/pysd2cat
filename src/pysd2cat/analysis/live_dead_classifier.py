from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
import pandas as pd
from pysd2cat.data import pipeline

#from test_harness.test_harness_class import TestHarness
#from test_harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification



def build_model_pd(df):
    print("Length of full DF", len(df))
    input_cols = ['FSC-A', 'SSC-A', 'BL1-A', 'RL1-A', 'FSC-H', 'SSC-H', 'BL1-H', 'RL1-H', 'FSC-W', 'SSC-W', 'BL1-W', 'RL1-W']
    output_cols = ["class_label"]
    print("Size of filtered data", len(df))
    train, test = train_test_split(df, stratify=df['class_label'], test_size=0.2, random_state=5)
    th = TestHarness(output_path='harness_results')

    rf_classification_model = random_forest_classification(n_estimators=500)
    th.add_custom_runs(test_harness_models=rf_classification_model, 
                       training_data=train, 
                       testing_data=test,
                       data_and_split_description="yeast_live_dead_dataframe",
                       cols_to_predict=output_cols,
                       feature_cols_to_use=input_cols, 
                       normalize=True, 
                       feature_cols_to_normalize=input_cols,
                       feature_extraction='rfpimp_permutation', 
                       predict_untested_data=False)
    # Mohammed add end
    th.execute_runs()


def build_model(dataframe):
    #print(c_df_norm[0:5,:])
    #df_norm = dataframe
    
    X = dataframe.drop(columns=['class_label'])
    y = dataframe['class_label'].astype(int)

    #binarizer = Binarizer(threshold=0.0).fit(y)
    #y = binarizer.transform(y)[:,0]

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

    scaler = Normalizer().fit(train_X)
    train_X_norm = scaler.transform(train_X)


    
    #print(c_df.columns[1:12])
    #val_X = pd.DataFrame(val_X, columns = dataframe.columns[1:13])
    #val_y = pd.DataFrame(val_y, columns = [dataframe.columns[0]])

    # Define model
    #logreg = LogisticRegression()
    rf_model = RandomForestClassifier(random_state=1, class_weight = 'balanced')

    # Fit model
    #logreg.fit(train_X, train_y)
    rf_model.fit(train_X_norm, train_y)

    test_scaler = Normalizer().fit(train_X)
    val_X_norm = test_scaler.transform(val_X)
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
    df_norm = scaler.transform(df)
    predictions = model.predict(df_norm)
    #predictions = model.predict(df)
    df['class_label'] = pd.DataFrame(predictions, columns = ['class_label'])
    return df
    