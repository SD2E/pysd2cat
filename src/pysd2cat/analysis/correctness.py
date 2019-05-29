from harness.test_harness_class import TestHarness
from harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification
from pysd2cat.analysis.Names import Names
from pysd2cat.analysis import live_dead_classifier as ldc
import numpy as np
from pysd2cat.analysis.threshold import compute_correctness
import pandas as pd
import logging

l = logging.getLogger(__file__)
l.setLevel(logging.DEBUG)


def strain_to_class(x,  high_control=Names.NOR_00_CONTROL, low_control=Names.WT_LIVE_CONTROL):
    if x['strain_name'] == low_control:
        return 0
    elif x['strain_name'] == high_control:
        return 1
    else:
        return None


def get_classifier_dataframe(df, data_columns = ['FSC_A', 'SSC_A', 'BL1_A', 'RL1_A', 'FSC_H', 'SSC_H', 'BL1_H', 'RL1_H', 'FSC_W', 'SSC_W', 'BL1_W', 'RL1_W'],
                             high_control=Names.NOR_00_CONTROL, low_control=Names.WT_LIVE_CONTROL):
    """
    Get the classifier data corresponding to controls.
    """
    low_df = df.loc[df['strain_name'] == low_control]
    high_df = df.loc[df['strain_name'] == high_control]
    low_df.loc[:,'strain_name'] = low_df.apply(lambda x : strain_to_class(x,  high_control=high_control, low_control=low_control), axis=1)
    high_df.loc[:,'strain_name'] = high_df.apply(lambda x : strain_to_class(x,  high_control=high_control, low_control=low_control), axis=1)
    low_high_df = low_df.append(high_df)

    #print(live_dead_df)
    low_high_df = low_high_df.rename(index=str, columns={'strain_name': "class_label"})
    low_high_df = low_high_df[data_columns + ['class_label']]
    return low_high_df

def compute_predicted_output_harness(df, data_columns = ['FSC_A', 'SSC_A', 'BL1_A', 'RL1_A', 'FSC_H', 'SSC_H', 'BL1_H', 'RL1_H', 'FSC_W', 'SSC_W', 'BL1_W', 'RL1_W'], out_dir='.',
                                     high_control=Names.NOR_00_CONTROL, low_control=Names.WT_LIVE_CONTROL):
    ## Build the training/test input
    l.debug("Building Train/Test data from controls ...")
    c_df = get_classifier_dataframe(df, data_columns = data_columns, high_control=high_control, low_control=low_control)
    
    ## Build the classifier
    ## Predict label for unseen data
    l.debug("Building Classifier ...")
    pred_df = ldc.build_model_pd(c_df,
                                 data_df = df,
                                 input_cols=data_columns,
                                 output_cols=['class_label'],
                                 output_location=out_dir,
                                 description="circuit output prediction")
    #print(pred_df)
    result_df = df[['output', 'id']]
    l.debug("Predictions: " + str(pred_df['class_label_predictions']))
    #print(result_df)
    result_df.loc[:,'predicted_output'] = pred_df['class_label_predictions'].astype(int)
    l.debug("predicted_output vcs: " + str(result_df['predicted_output'].value_counts()))
    l.debug("output vcs: " + str(result_df['output'].value_counts())) 
    #print(df)
    
    return result_df

def compute_correctness_harness(df,
                             out_dir = '.',
                             mean_output_label='probability_correct',
                             std_output_label='probability_correct',                             
                             high_control=Names.NOR_00_CONTROL,
                             low_control=Names.WT_LIVE_CONTROL):
    df.loc[:,'output'] = pd.to_numeric(df['output'])
    l.debug("Shape at start: " + str(df.shape))
    result_df = compute_predicted_output_harness(df, out_dir=out_dir, high_control=high_control, low_control=low_control)
    l.debug("Shape of result: " + str(result_df.shape) + ", columns= " + str(result_df.columns))
    result_df.loc[:, 'predicted_correct'] = result_df.apply(lambda x : None if np.isnan(x['output']) or np.isnan(x['predicted_output']) else 1.0 - np.abs(x['output'] - x['predicted_output']), axis=1)
    l.debug(result_df)
    #acc_df = result_df.groupby(['id'])['predicted_correct'].agg([np.nanmean, np.nanstd]).reset_index()
    
    def nan_agg(x):
        res = {}
        x = x.dropna()
        if len(x) > 0:
            res['mean'] = x['predicted_correct'].mean()
            res['std'] = x['predicted_correct'].std()
        else:
            res['mean'] = None
            res['std'] = None

        return pd.Series(res, index=['mean', 'std'])

    groups = result_df.groupby(['id'])
    acc_df = groups.apply(nan_agg).reset_index() 

    l.debug("Shape of acc: " + str(acc_df.shape))
    #print(acc_df)
    acc_df = acc_df.rename(columns={'mean' : mean_output_label, 'std' : std_output_label })
    #print(acc_df)
    return acc_df
    

def compute_correctness_all(df, out_dir = '.', high_control=Names.NOR_00_CONTROL, low_control=Names.WT_LIVE_CONTROL):
    drop_list = ['Time', 'FSC_A', 'SSC_A', 'BL1_A', 'RL1_A', 'FSC_H', 'SSC_H',
                      'BL1_H', 'RL1_H', 'FSC_W', 'SSC_W', 'BL1_W', 'RL1_W', 'live']
    if 'index' in df.columns:
        drop_list.append('index')
    result = df.drop(drop_list,
                      axis=1).drop_duplicates().reset_index()


    results = []
    # Get correctness w/o dead cells gated
    l.debug("Computing Threhold, no gating ...")
    results.append(compute_correctness(df,
                                    output_label='probability_correct_threshold',
                                    high_control=high_control,
                                    low_control=low_control,
                                    mean_name='mean_log_gfp',
                                    std_name='std_log_gfp',
                                    mean_correct_name='mean_correct_threshold',
                                    std_correct_name='std_correct_threshold',
                                    count_name='count',
                                    threshold_name='threshold'))

    l.debug("Computing Random Forest, no gating ...")
    results.append(compute_correctness_harness(df,
                                            out_dir = out_dir,
                                            mean_output_label='mean_correct_classifier',
                                            std_output_label='std_correct_classifier',                                            
                                            high_control=high_control,
                                            low_control=low_control))
    
    # Get correctness w/ dead cells gated
    if 'live' in df.columns:
        gated_df = df.loc[df['live'] == 1]
        l.debug("Computing Threhold, with gating ...")
        results.append(compute_correctness(gated_df,
                                        output_label='probability_correct_threshold',
                                        high_control=high_control,
                                        low_control=low_control,
                                        mean_name='mean_log_gfp_live',
                                        std_name='std_log_gfp_live',
                                        mean_correct_name='mean_correct_threshold_live',
                                        std_correct_name='std_correct_threshold_live',
                                        count_name='count_live',
                                        threshold_name='threshold_live'))

        l.debug("Computing Random Forest, with gating ...")
        results.append(compute_correctness_harness(gated_df,
                                                out_dir = out_dir,
                                                mean_output_label='mean_correct_classifier_live',
                                                std_output_label='std_correct_classifier_live',                                            
                                                high_control=high_control,
                                                low_control=low_control))

    #print(len(result))
    for r in results:
        result = result.merge(r, on='id')
        #print(len(result))
    return result
