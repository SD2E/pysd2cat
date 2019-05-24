import pandas as pd
import math
import numpy as np
import os
from pysd2cat.data import pipeline
from pysd2cat.analysis.Names import Names    

def write_correctness(data_file, overwrite, high_control=Names.NOR_00_CONTROL, low_control=Names.WT_LIVE_CONTROL):
    try:
        data_dir = "/".join(data_file.split('/')[0:-1])
        out_path = os.path.join(data_dir, 'accuracy')
        out_file = os.path.join(out_path, data_file.split('/')[-1])
        if overwrite or not os.path.isfile(out_file):
            print("Computing correctness file: " + out_file)
            data_df = pd.read_csv(data_file,memory_map=True,dtype={'od': float, 'input' : object, 'output' : object}, index_col=0 )
            correctness_df = compute_correctness(data_df, high_control=high_control, low_control=low_control)
            print("Writing correctness file: " + out_file)
            correctness_df.to_csv(out_file)
    except Exception as e:
        print("File failed: " + data_file + " with: " + str(e))
        pass


def write_correctness_files(data, overwrite=False, high_control=Names.NOR_00_CONTROL, low_control=Names.WT_LIVE_CONTROL):
    import multiprocessing
    pool = multiprocessing.Pool(int(multiprocessing.cpu_count()))
    multiprocessing.cpu_count()
    tasks = []
    for d in data:
#        tasks.append((d, overwrite, high_control=high_control, low_control=low_control))
        tasks.append((d, overwrite))
    results = [pool.apply_async(write_correctness, t) for t in tasks]

    for result in results:
        data_list = result.get()
 

def get_experiment_correctness_and_metadata(adf):
    """
    Cleanup the correctness dataframe from an experiment.
    """
    def media_fix(x):
        media_map = {
            "SC Media" : "SC Media",
            "SC+Oleate+Adenine" : "SC Media",
            "standard_media" : "SC Media",
            "Synthetic_Complete" : "SC Media",

            "Synthetic_Complete_2%Glycerol_2%Ethanol" : "SC Slow",
            "SC Slow" : "SC Slow",
            "slow_media" : "SC Slow",



            "Synthetic_Complete_1%Sorbitol" : "SC High Osm",
            "SC High Osm" : "SC High Osm",    
            'high_osm_media' : "SC High Osm",    

            "YPAD" : "SC Rich",
            "Yeast_Extract_Peptone_Adenine_Dextrose (a.k.a. YPAD Media)" : "SC Rich",
            "rich_media" : "SC Rich",
        }
        if type(x) is str:
            return media_map[x]
        else:
            return x
        
    def fix_temp(x):
        if type(x['inc_temp']) is str:
            x['inc_temp'] = float(x['inc_temp'].split("_")[1])
        return x

    def fix_time(x):
        if 'inc_time_2' not in x or x['inc_time_2'] is None:
            x['inc_time_2'] = 18
        elif type(x['inc_time_2']) is str:
            x['inc_time_2'] = float(x['inc_time_2'].split(":")[0])

        return x


    
    drop_list = ['Unnamed: 0', 'FSC_A',
           'SSC_A', 'BL1_A', 'RL1_A', 'FSC_H', 'SSC_H', 'BL1_H', 'RL1_H', 'FSC_W',
           'SSC_W', 'BL1_W', 'RL1_W', 'Time']
    final_df = adf
    #print(final_df['media'])
    final_df.loc[:, 'media'] = final_df['media'].apply(media_fix)
    final_df = final_df.apply(fix_input, axis=1)            
    final_df = final_df.apply(fix_output, axis=1)
    final_df = final_df.apply(fix_temp, axis=1)
    final_df = final_df.apply(fix_time, axis=1)


    #print(final_df['media'])
    
    return final_df

def get_sample_correctness(data):
    """
    Get a dataframe that includes the correctness of all samples and condition sets.
    Requires dataframes for correctness are present in paths listed in 'data' parameter.
    """
    all_df = pd.DataFrame()
    for d in data:
        if os.path.isfile(d):
            #print("Getting Correctness for: " + str(d))
            #df = pd.read_csv(d)
            adata = os.path.join("/".join(d.split('/')[0:-1]), 'accuracy', d.split('/')[-1])
            if os.path.isfile(adata):
                adf = pd.read_csv(adata, dtype={'od': float, 'input' : object}, index_col=0)
                if 'media' in adf.columns:
                    final_df = get_experiment_correctness_and_metadata(adf)
                    all_df = all_df.append(final_df, ignore_index=True)
    all_df.loc[:, 'prc_improve'] = all_df['probability_correct_live'] - all_df['probability_correct']
    all_df.loc[:, 'live_proportion'] = all_df['count_live'] / all_df['count']
    #all_df['sample_time'] = all_df.apply(pipeline.get_sample_time, axis=1)
    return all_df




def get_threshold(df, channel='BL1_A', high_control=Names.NOR_00_CONTROL, low_control=Names.WT_LIVE_CONTROL):

    ## Prepare the data for high and low controls
    high_df = df.loc[( df['strain_name'] == high_control)]
    high_df.loc[:,'output'] = high_df.apply(lambda x: 1, axis=1)
    low_df = df.loc[(df['strain_name'] == low_control) ]
    low_df.loc[:,'output'] = low_df.apply(lambda x: 0, axis=1)
    high_low_df = high_df.append(low_df)
    high_low_df = high_low_df.loc[high_low_df[channel] > 0]
    high_low_df.loc[:, channel] = np.log(high_low_df[channel]).replace([np.inf, -np.inf], np.nan).dropna()
    #high_low_df[channel]
    
    if len(high_df) == 0 or len(low_df) == 0:
        raise Exception("Cannot compute threshold if do not have both low and high control")

    ## Setup Gradient Descent Paramters

    cur_x = high_low_df[channel].mean() # The algorithm starts at mean
    #print("Starting theshold = " + str(cur_x))
    rate = 0.00001 # Learning rate
    precision = 0.0001 #This tells us when to stop the algorithm
    previous_step_size = 1 #
    max_iters = 100 # maximum number of iterations
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


def fix_input(row):
    if row['input'] == '1.0':
        row['input'] = '01'
    elif row['input'] == '0.0':
        row['input'] = '00'
    elif row['input'] == '10.0':
        row['input'] = '10'
    elif row['input'] == '11.0':
        row['input'] = '11'
    return row

def fix_output(row):
    if row['output'] == '1.0':
        row['output'] = '1'
    elif row['output'] == '0.0':
        row['output'] = '0'
    return row

def compute_correctness(m_df,
                     channel='BL1_A',
                     thresholds=None,
                     use_log_value=True,
                     high_control=Names.NOR_00_CONTROL,
                     low_control=Names.WT_LIVE_CONTROL,
                     output_label='probability_correct',
                     mean_name='mean_log_gfp',
                     std_name='std_log_gfp',
                     mean_correct_name='probability_correct',
                     std_correct_name='std_correct',
                     count_name='count',
                     threshold_name='threshold'
                     ):
    if thresholds is None:
        try:
            threshold, threshold_quality = get_threshold(m_df, channel, high_control=high_control, low_control=low_control)
            thresholds = [threshold]
        except Exception as e:
            #print(e)
            raise Exception("Could not find controls to auto-set threshold: " + str(e))
            #thresholds = [np.log(10000)]
      
    #print("Threshold  = " + str(thresholds[0]))
    samples = m_df.groupby(['id'])
    plot_df = pd.DataFrame()
    for id, sample in samples:
        #print(sample_id)
        #sample = m_df.loc[m_df['id'] == sample_id]
        #print(sample.head())
        circuit = sample['gate'].dropna().unique()
        if len(circuit) == 1:
            value_df = sample[[channel, 'output']].rename(index=str, columns={channel: "value"})          
            if use_log_value:
                value_df = value_df.loc[value_df['value'] > 0]
                value_df.loc[:,'value'] = np.log(value_df['value']).replace([np.inf, -np.inf], np.nan).dropna()
            #print(value_df.head())
            thold_df = do_threshold_analysis(value_df,
                                             thresholds,
                                             mean_correct_name=mean_correct_name,
                                             std_correct_name=std_correct_name,
                                             count_name=count_name,
                                             threshold_name=threshold_name)

            
            thold_df[mean_name] = np.mean(value_df['value'])
            thold_df[std_name] = np.std(value_df['value'])
            
            thold_df['id'] = id
#            for i in ['gate', 'input', 'output', 'od', 'media',
#                      'inc_temp', 'replicate', 'inc_time_1',
#                      'inc_time_2', 'strain_name']:
#                if i in sample.columns:
#                    thold_df[i] = sample[i].unique()[0]
#                elif i == 'inc_temp':
#                    thold_df[i] = 'warm_30'
#                else:
#                    thold_df[i] = None
#            thold_df = thold_df.apply(fix_input, axis=1)            
#            thold_df = thold_df.apply(fix_output, axis=1)
            
            ## if 'live' in m_df.columns:
            ##     sample_live = sample.loc[sample['live'] == 1]
            ##     value_df = sample_live[[channel, 'output']].rename(index=str, columns={channel: "value"})
            ##     if use_log_value:
            ##         value_df = value_df.loc[value_df['value'] > 0]
            ##         value_df['value'] = np.log(value_df['value']).replace([np.inf, -np.inf], np.nan).dropna()

            ##     #print(value_df.shape())
            ##     thold_live_df = do_threshold_analysis(value_df, thresholds)
            ##     thold_df['probability_correct_live'] = thold_live_df['probability_correct']
            ##     thold_df['standard_error_correct_live'] = thold_live_df['standard_error_correct']
            ##     thold_df['count_live'] = thold_live_df['count']
            ##     thold_df['mean_log_gfp_live'] = np.mean(value_df['value'])
            ##     thold_df['std_log_gfp_live'] = np.std(value_df['value'])
            ## else:
            ##     thold_df['probability_correct_live'] = None #thold_df['probability_correct']
            ##     thold_df['standard_error_correct_live'] = None #thold_df['standard_error_correct']
            ##     thold_df['count_live'] = None #thold_df['count']
            ##     thold_df['mean_log_gfp_live'] = None
            ##     thold_df['std_log_gfp_live'] = None



                
            #print(thold_df)
            plot_df = plot_df.append(thold_df, ignore_index=True)
    #plot_df = plot_df.rename(columns={mean_correct_name : output_label})
    return plot_df 

def do_threshold_analysis(df,
                          thresholds,
                          mean_correct_name='probability_correct',
                          std_correct_name='std_correct',
                          count_name='count',
                          threshold_name='threshold'
                          ):
    """
    Get Probability that samples fall on correct side of threshold
    """
    count = 0
    correct = []
    for idx, threshold in enumerate(thresholds):
        correct.append(0)
        
    for idx, row in df.iterrows():
        true_gate_output = int(row['output'])
        measured_gate_output = float(row['value'])
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
            mean_correct_name : pr, 
            std_correct_name : se,
            count_name : count,
            threshold_name : threshold}, ignore_index=True)
    return results
    

    

