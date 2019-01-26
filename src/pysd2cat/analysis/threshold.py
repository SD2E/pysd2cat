import pandas as pd
import math
import numpy as np
import os
from pysd2cat.data import pipeline

def get_experiment_accuracy_and_metadata(adf):
    """
    Cleanup the accuracy dataframe from an experiment.
    """
    def media_fix(x):
        media_map = {
            "SC Media" : "SC Media",
            "SC+Oleate+Adenine" : "SC Media",
            "standard_media" : "SC Media",

            "Synthetic_Complete_2%Glycerol_2%Ethanol" : "SC Slow",
            "SC Slow" : "SC Slow",
            "slow_media" : "SC Slow",



            "Synthetic_Complete_1%Sorbitol" : "SC High Osm",
            "SC High Osm" : "SC High Osm",    
            'high_osm_media' : "SC High Osm",    

            "YPAD" : "YPAD",
            "Yeast_Extract_Peptone_Adenine_Dextrose (a.k.a. YPAD Media)" : "YPAD",
            "rich_media" : "YPAD",
        }
        if type(x) is str:
            return media_map[x]
        else:
            return x

    drop_list = ['Unnamed: 0', 'FSC_A',
           'SSC_A', 'BL1_A', 'RL1_A', 'FSC_H', 'SSC_H', 'BL1_H', 'RL1_H', 'FSC_W',
           'SSC_W', 'BL1_W', 'RL1_W', 'Time']
    final_df = adf
    final_df['media'] = final_df['media'].apply(media_fix)
    return final_df

def get_sample_accuracy(data):
    """
    Get a dataframe that includes the accuracy of all samples and condition sets.
    Requires dataframes for accuracy are present in paths listed in 'data' parameter.
    """
    all_df = pd.DataFrame()
    for d in data:
        if os.path.isfile(d):
            #print("Getting Accuracy for: " + str(d))
            #df = pd.read_csv(d)
            adata = os.path.join("/".join(d.split('/')[0:-1]), 'accuracy', d.split('/')[-1])
            if os.path.isfile(adata):
                adf = pd.read_csv(adata, dtype={'od': float, 'input' : object})
                if 'media' in adf.columns:
                    final_df = get_experiment_accuracy_and_metadata(adf)
                    all_df = all_df.append(final_df)
    all_df['prc_improve'] = all_df['probability_correct_live'] - all_df['probability_correct']
    all_df['live_proportion'] = all_df['count_live'] / all_df['count']
    #all_df['sample_time'] = all_df.apply(pipeline.get_sample_time, axis=1)
    return all_df




def get_threshold(df, channel='BL1_A'):

    ## Prepare the data for high and low controls
    high_df = df.loc[( df['strain_name'] == 'NOR-00-Control')]
    high_df['output'] = 1
    low_df = df.loc[(df['strain_name'] == 'WT-Live-Control') ]
    low_df['output'] = 0
    high_low_df = high_df.append(low_df)
    high_low_df = high_low_df.loc[high_low_df[channel] > 0]
    high_low_df[channel] = np.log(high_low_df[channel]).replace([np.inf, -np.inf], np.nan).dropna()
    high_low_df[channel]

    ## Setup Gradient Descent Paramters

    cur_x = high_low_df[channel].mean() # The algorithm starts at mean
    rate = 0.00001 # Learning rate
    precision = 0.000001 #This tells us when to stop the algorithm
    previous_step_size = 1 #
    max_iters = 10000 # maximum number of iterations
    iters = 0 #iteration counter

    def correct_side(threshold, value, output):
        if output == 1 and value > threshold:
            return 1
        elif output == 0 and value <= threshold:
            return 1
        else:
            return 0

    def gradient(x):
        delta = 0.1
        xp = x + delta
        correct = high_low_df.apply(lambda row : correct_side(x, row['BL1_A'], row['output']), axis=1)
        correctp = high_low_df.apply(lambda row : correct_side(xp, row['BL1_A'], row['output']), axis=1)
        # print(sum(correct))
        # print(sum(correctp))
        try:
            grad = (np.sum(correct) - np.sum(correctp))/delta
        except Exception as e:
            print(sum(correct))
            print(sum(correctp))
            print(e)
        #print("Gradient at: " + str(x) + " is " + str(grad))
        return grad

    while previous_step_size > precision and iters < max_iters:
        prev_x = cur_x #Store current x value in prev_x
        cur_x = cur_x - rate * gradient(prev_x) #Grad descent
        previous_step_size = abs(cur_x - prev_x) #Change in x
        iters = iters+1 #iteration count
        #print("Iteration",iters,"\nX value is",cur_x) #Print iterations

    max_value = sum(high_low_df.apply(lambda row : correct_side(cur_x, row['BL1_A'], row['output']), axis=1))/len(high_low_df)
    # print("Maximized at: " + str(cur_x))
    # print("Value at Max: " + str(max_value))
    return cur_x, max_value

def compute_accuracy(m_df, channel='BL1_A', thresholds=None, use_log_value=True):
    if thresholds is None:
        try:
            threshold, threshold_quality = get_threshold(m_df, channel)
            thresholds = [threshold]
        except Exception as e:
            print("Could not find controls to auto-set threshold")
            thresholds = [np.log(10000)]
      
        
    samples = m_df['id'].unique()
    plot_df = pd.DataFrame()
    for sample_id in samples:
        #print(sample_id)
        sample = m_df.loc[m_df['id'] == sample_id]
        #print(sample.head())
        circuit = sample['gate'].unique()[0]
        if type(circuit) is str:
            #print(output)
            value_df = sample[[channel, 'output']].rename(index=str, columns={channel: "value"})          
            if use_log_value:
                value_df = value_df.loc[value_df['value'] > 0]
                value_df['value'] = np.log(value_df['value']).replace([np.inf, -np.inf], np.nan).dropna()
            #print(value_df.shape())
            thold_df = do_threshold_analysis(value_df, thresholds)
            
            thold_df['id'] = sample_id
            for i in ['gate', 'input', 'od', 'media', 'inc_temp']:
                if i in sample.columns:
                    thold_df[i] = sample[i].unique()[0]
                elif i == 'inc_temp':
                    thold_df[i] = 'warm_30'
                else:
                    thold_df[i] = None
            
            if 'live' in m_df.columns:
                sample_live = sample.loc[sample['live'] == 1]
                value_df = sample_live[[channel, 'output']].rename(index=str, columns={channel: "value"})
                if use_log_value:
                    value_df = value_df.loc[value_df['value'] > 0]
                    value_df['value'] = np.log(value_df['value']).replace([np.inf, -np.inf], np.nan).dropna()

                #print(value_df.shape())
                thold_live_df = do_threshold_analysis(value_df, thresholds)
                thold_df['probability_correct_live'] = thold_live_df['probability_correct']
                thold_df['standard_error_correct_live'] = thold_live_df['standard_error_correct']
                thold_df['count_live'] = thold_live_df['count']
            else:
                thold_df['probability_correct_live'] = thold_df['probability_correct']
                thold_df['standard_error_correct_live'] = thold_df['standard_error_correct']
                thold_df['count_live'] = thold_df['count']


            #print(thold_df)
            plot_df = plot_df.append(thold_df, ignore_index=True)
    return plot_df 

def do_threshold_analysis(df, thresholds):
    """
    Get Probability that samples fall on correct side of threshold
    """
    count = 0
    correct = []
    for idx, threshold in enumerate(thresholds):
        correct.append(0)
        
    for idx, row in df.iterrows():
        true_gate_output = row['output']
        measured_gate_output = row['value']
        count = count + 1
        for idx, threshold in enumerate(thresholds):
            #print(str(true_gate_output) + " " + str(measured_gate_output))
            if (true_gate_output == 1 and measured_gate_output >= threshold) or \
               (true_gate_output == 0 and measured_gate_output < threshold) :
                correct[idx] = correct[idx] + 1
            
    results = pd.DataFrame()
    
    for idx, threshold in enumerate(thresholds):
        if count > 0:
            pr = correct[idx] / count
            se = math.sqrt(pr*(1-pr)/count)
        else:
            pr = 0
            se = 0

        results= results.append({
            'probability_correct' : pr, 
            'standard_error_correct' : se,
            'count' : count,
            'threshold' : threshold}, ignore_index=True)
    return results
    

    

