import pandas as pd
import math


def compute_accuracy(m_df, channel='BL1_A', thresholds=[10000]):
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
            #print(value_df.shape())
            thold_df = do_threshold_analysis(value_df, thresholds)
            thold_df['id'] = sample_id
            for i in ['gate', 'input', 'od', 'media', 'inc_temp']:
                thold_df[i] = sample[i].unique()[0]
            
            if 'live' in m_df.columns:
                sample_live = sample.loc[sample['live'] == 1]
                value_df = sample_live[[channel, 'output']].rename(index=str, columns={channel: "value"})
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
    

    

