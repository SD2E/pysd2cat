import pandas as pd
import math
import numpy as np


def get_threshold(df, channel='BL1_A'):

    ## Prepare the data for high and low controls
    high_df = df.loc[( df['strain_name'] == 'NOR-00-Control')]
    high_df['output'] = 1
    low_df = df.loc[(df['strain_name'] == 'WT-Live-Control') ]
    low_df['output'] = 0
    high_low_df = high_df.append(low_df)
    high_low_df[channel] = np.log(high_low_df[channel]).replace([np.inf, -np.inf], np.nan).dropna()
    high_low_df[channel]

    ## Setup Gradient Descent Paramters

    cur_x = high_low_df[channel].mean() # The algorithm starts at mean
    rate = 0.0001 # Learning rate
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
        grad = (sum(correct) - sum(correctp))/delta
        # print("Gradient at: " + str(x) + " is " + str(grad))
        return grad

    while previous_step_size > precision and iters < max_iters:
        prev_x = cur_x #Store current x value in prev_x
        cur_x = cur_x - rate * gradient(prev_x) #Grad descent
        previous_step_size = abs(cur_x - prev_x) #Change in x
        iters = iters+1 #iteration count
        # print("Iteration",iters,"\nX value is",cur_x) #Print iterations

    max_value = sum(high_low_df.apply(lambda row : correct_side(cur_x, row['BL1_A'], row['output']), axis=1))/len(high_low_df)
    # print("Maximized at: " + str(cur_x))
    # print("Value at Max: " + str(max_value))
    return cur_x, max_value

def compute_accuracy(m_df, channel='BL1_A', thresholds=None, use_log_value=True):
    if thresholds is None:
        threshold, threshold_quality = get_threshold(m_df, channel)
        thresholds = [threshold]
        
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
                value_df['value'] = np.log(value_df['value']).replace([np.inf, -np.inf], np.nan).dropna()
            #print(value_df.shape())
            thold_df = do_threshold_analysis(value_df, thresholds)
            thold_df['id'] = sample_id
            for i in ['gate', 'input', 'od', 'media', 'inc_temp']:
                thold_df[i] = sample[i].unique()[0]
            
            if 'live' in m_df.columns:
                sample_live = sample.loc[sample['live'] == 1]
                value_df = sample_live[[channel, 'output']].rename(index=str, columns={channel: "value"})
                if use_log_value:
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
    

    

