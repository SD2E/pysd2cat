import pymongo
import json
import pandas as pd
import os
import FlowCytometryTools as FCT
import numpy as np

***REMOVED***
client = pymongo.MongoClient(dbURI)
db = client.catalog
science_table=db.science_table

###############################################
# Helpers for building a live/dead classifier #
###############################################

def get_dataframe_for_live_dead_classifier(data_dir):
    """
    Get pooled FCS data for every live and dead control. 
    """
    meta_df = get_metadata_dataframe(get_live_dead_controls())
    
    ##Drop columns that we don't need
    da = meta_df[['strain', 'filename']].copy()
    da['strain'] = da['strain'].mask(da['strain'] == 'WT-Dead-Control',  0)
    da['strain'] = da['strain'].mask(da['strain'] == 'WT-Live-Control',  1)
    da = da.rename(index=str, columns={"strain": "class_label"})
    da = get_data_and_metadata_df(da, data_dir)
    da = da.drop(columns=['filename', 'Time'])
    return da

def get_live_dead_controls():
    """
    Get metadata for every live and dead control sample across
    all experiments.
    """
    query={}
    query['challenge_problem'] = 'YEAST_GATES'
    query['file_type'] = 'FCS'
    query['strain'] = {"$in": ['WT-Dead-Control', 'WT-Live-Control']}

    results = []
    for match in science_table.find(query):
        match.pop('_id')
        results.append(match)
    return results


def get_metadata_dataframe(results):
    """
    Convert science table results into metadata dataframe.
    """
    meta_df = pd.DataFrame()
    for result in results:
        result_df = {}
        keys_to_set = ['strain', 'filename', 'lab', 'sample_id',
                       'strain_circuit', 'strain_input_state', 
                       'strain_sbh_uri', 'experiment_id'
                      ]
        for k in keys_to_set:
            if k in result:
                result_df[k] = result[k]
            else:
                result_df[k] = None

        ## Other values (not at top level)
        if 'inoculation_density' in result:
            result_df['od'] = result['inoculation_density']['value']
        else:
            result_df['od'] = None
        if 'sample_contents' in result:
            result_df['media'] = result['sample_contents'][0]['name']['sbh_uri']
        else:
            result_df['media'] = None
            
        if 'strain_circuit' in result and 'strain_input_state' in result:
            result_df['output'] = gate_output(result['strain_circuit'], result['strain_input_state'])
        else:
            result_df['output'] = None

        #result_df['fcs_files'] = ['agave://' + result['agave_system'] + result['agave_path']]

#        try:
#            result_ex_id = result['sample_id'].split("/")[0].split(".")[-1]
#        except Exception as e:
#            result_ex_id = None
#        result_df['ex_id'] = result_ex_id


        meta_df = meta_df.append(result_df, ignore_index=True)
    #pd.set_option('display.max_colwidth', -1)    
    return meta_df


def get_data_and_metadata_df(metadata_df, data_dir, fraction=None):
    """
    Join each FCS datatable with its metadata.  Costly!
    """
    #dataset_local_df=pd.DataFrame()
    all_data_df = pd.DataFrame()
    for i, record in metadata_df.iterrows():
        ## Substitute local file for SD2 URI to agave file 
        #record['fcs_files'] = local_datafile(record['fcs_files'][0], data_dir)
        #dataset_local_df = dataset_local_df.append(record)
    
        ## Create a data frame out of FCS file
        data_df = FCT.FCMeasurement(ID=record['filename'],
                                    datafile=os.path.join(data_dir, record['filename'])).read_data()
        if fraction is not None:
            data_df = data_df.sample(frac=fraction, replace=True)
            #data_df = data_df.replace([np.inf, -np.inf], np.nan)
        #data_df = data_df[~data_df.isin(['NaN', 'NaT']).any(axis=1)]
        
        data_df['filename'] = record['filename']
        all_data_df = all_data_df.append(data_df)

    ## Join data and metadata
    final_df = metadata_df.set_index('filename').join(all_data_df.set_index('filename'))
    final_df = final_df.reset_index()
    final_df = final_df.dropna()
    
    return final_df


###############################################
# Helpers for getting sample data to classify #
###############################################

def gate_output(gate, inputs):
    if gate == 'NOR':
        if inputs == '00':
            return 1
        else:
            return 0
    elif gate == 'AND':
        if inputs == '11':
            return 1
        else:
            return 0
    elif gate == 'NAND':
        if inputs == '11':
            return 0
        else:
            return 1
    elif gate == 'OR':
        if inputs == '00':
            return 0
        else:
            return 1
    elif gate == 'XOR':
        if inputs == '00' or inputs == '11':
            return 0
        else:
            return 1
    elif gate == 'XNOR':
        if inputs == '00' or inputs == '11':
            return 1
        else:
            return 0

def get_strain_dataframe_for_classifier(circuit, input, od=0.0003, media='SC Media', experiment='', data_dir='', fraction=None):
    """
    """
    meta_df = get_metadata_dataframe(get_strain(circuit, input, od=od, media=media, experiment=experiment))
    da = meta_df[['filename', 'output']].copy()
    #da['strain'] = da['strain'].mask(da['strain'] == 'WT-Dead-Control',  0)
    #da['strain'] = da['strain'].mask(da['strain'] == 'WT-Live-Control',  1)
    da = get_data_and_metadata_df(da, data_dir, fraction=fraction)
    #da = da.drop(columns=['fcs_files', 'Time', 'RL1-W', 'BL1-W'])
    da = da.drop(columns=['filename', 'Time'])
    return da


def get_strain(strain_circuit, strain_input_state,od=0.0003, media='SC Media',experiment=''):
    query={}
    query['challenge_problem'] = 'YEAST_GATES'
    query['file_type'] = 'FCS'
    query['lab'] = 'Transcriptic'
    
#    if strain_circuit == 'XOR' and strain_input_state == '00':
#        query['strain'] = '16970'
#    else:
    query['strain_circuit'] = strain_circuit
    query['strain_input_state'] = strain_input_state
    query['inoculation_density.value'] =  od
    query['sample_contents.0.name.label'] =  media

    #    query['experiment_id'] = experiment
    #query['filename'] = { "$regex" : ".*" + experiment +".*"}
    
    #query['strain'] = {"$in": ['WT-Dead-Control', 'WT-Live-Control']}

    results = []
    for match in science_table.find(query):
        match.pop('_id')
        results.append(match)
    #dumpfile = "strains.json"
    #json.dump(results,open(dumpfile,'w'))
    return results









def get_control(circuit, control, od=0.0003, media='SC Media'):
    query={}
    query['challenge_problem'] = 'YEAST_GATES'
    query['file_type'] = 'FCS'
    query['lab'] = 'Transcriptic'
    #query['strain_circuit'] = strain_circuit
    #query['strain_input_state'] = strain_input_state
    query['inoculation_density.value'] =  od
    query['sample_contents.0.name.label'] =  media
    
    query['strain'] = control

    results = []
    for match in science_table.find(query):
        match.pop('_id')
        results.append(match)
    dumpfile = "strains.json"
    json.dump(results,open(dumpfile,'w'))
    return results

def get_experiment_ids(challenge_problem='YEAST_GATES', lab='Transcriptic'):
    query={}
    query['challenge_problem'] = challenge_problem
    query['lab'] = lab

    results = []
    for match in science_table.find(query):
        match.pop('_id')
        results.append(match)
    #dumpfile = "strains.json"
    #json.dump(results,open(dumpfile,'w'))
    
    experiment_ids = list(frozenset([result['experiment_id'] for result in results]))
    
    return experiment_ids

def get_experiment_strains(experiment_id):
    query={}
    query['experiment_id'] = experiment_id

    results = []
    for match in science_table.find(query):
        match.pop('_id')
        results.append(match)
    #dumpfile = "strains.json"
    #json.dump(results,open(dumpfile,'w'))
    
    strains = list(frozenset([result['strain'] for result in results if 'strain' in result]))
   
    return strains

def get_fcs_measurements(strain, experiment_id):
    query={}
    query['strain'] = strain
    query['experiment_id'] = experiment_id
    query['file_type'] = 'FCS'


    results = []
    for match in science_table.find(query):
        match.pop('_id')
        results.append(match)
    #dumpfile = "strains.json"
    #json.dump(results,open(dumpfile,'w'))
    #print(results)
    files = list(frozenset([result['filename'] for result in results if 'filename' in result]))
   
    #result['agave_system'] + result['agave_path']
    return files





def get_control_dataframe_for_classifier(circuit, control, od=0.0003, media='SC Media', data_dir=''):
    meta_df = get_metadata_dataframe(get_control(circuit, control, od=od, media=media))
    da = meta_df[['fcs_files']].copy()
    #da['strain'] = da['strain'].mask(da['strain'] == 'WT-Dead-Control',  0)
    #da['strain'] = da['strain'].mask(da['strain'] == 'WT-Live-Control',  1)
    da = get_data_and_metadata_df(da, data_dir)
    #da = da.drop(columns=['fcs_files', 'Time', 'RL1-W', 'BL1-W'])
    da = da.drop(columns=['fcs_files', 'Time'])
    return da



## Convert agave path to local path
def local_datafile(datafile, data_dir):
    local_datafile = os.path.join(*(datafile.split('/')[3:]))   
    local_name = os.path.join(data_dir, local_datafile)
    print(local_name)
    return local_name


## Get one channel from FCS file
def get_measurements(fcs_file, channel, data_dir):
    #print("get_measurements: " + fcs_file + " " + channel)
    fct = FCT.FCMeasurement(ID=fcs_file, datafile=local_datafile(fcs_file, data_dir))
    return fct.data[[channel]]
    