import pymongo
import json
import pandas as pd
import os
import FlowCytometryTools as FCT
from pysd2cat.analysis.Names import Names

***REMOVED***
client = pymongo.MongoClient(dbURI)
db = client.catalog
science_table=db.science_table

###############################################
# Helpers for building a live/dead classifier #
###############################################

def get_dataframe_for_live_dead_classifier(data_dir,fraction=None, max_records=None):
    """
    Get pooled FCS data for every live and dead control. 
    """
    meta_df = get_metadata_dataframe(get_live_dead_controls())
    
    ##Drop columns that we don't need
    da = meta_df[[Names.STRAIN, Names.FILENAME]].copy()
    da[Names.STRAIN] = da[Names.STRAIN].mask(da[Names.STRAIN] == Names.WT_DEAD_CONTROL,  0)
    da[Names.STRAIN] = da[Names.STRAIN].mask(da[Names.STRAIN] == Names.WT_LIVE_CONTROL,  1)
    da = da.rename(index=str, columns={Names.STRAIN: "class_label"})
    da = get_data_and_metadata_df(da, data_dir,fraction,max_records)
    da = da.drop(columns=[Names.FILENAME, 'Time'])
    return da

def get_live_dead_controls():
    """
    Get metadata for every live and dead control sample across
    all experiments.
    """
    query={}
    query[Names.CHALLENGE_PROBLEM] = Names.YEAST_STATES
    query[Names.FILE_TYPE] = Names.FCS
    query[Names.STRAIN] = {"$in": [Names.WT_DEAD_CONTROL, Names.WT_LIVE_CONTROL]}

    results = []
    for match in science_table.find(query):
        match.pop('_id')
        results.append(match)
    return results


def get_experiment_samples(experiment_id, file_type):
    """
    Get metadata for every live and dead control sample across
    all experiments.
    """
    query={}
    query['experiment_id'] = experiment_id
    query['file_type'] = file_type

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
        keys_to_set = [Names.STRAIN, Names.FILENAME, Names.LAB, Names.SAMPLE_ID,
                       Names.STRAIN_CIRCUIT, Names.STRAIN_INPUT_STATE,
                       Names.STRAIN_SBH_URI, Names.EXPERIMENT_ID, Names.REPLICATE
                      ]
        for k in keys_to_set:
            if k in result:
                result_df[k] = result[k]
            else:
                result_df[k] = None

        ## Other values (not at top level)
        if Names.INOCULATION_DENSITY in result:
            result_df['od'] = result[Names.INOCULATION_DENSITY]['value']
        else:
            result_df['od'] = None
        if 'sample_contents' in result:
            result_df['media'] = result['sample_contents'][0]['name']['sbh_uri']
        else:
            result_df['media'] = None
            
        if Names.STRAIN_CIRCUIT in result and Names.STRAIN_INPUT_STATE in result:
            result_df['output'] = gate_output(result[Names.STRAIN_CIRCUIT], result[Names.STRAIN_INPUT_STATE])
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


def get_data_and_metadata_df(metadata_df, data_dir, fraction=None, max_records=None):
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
        data_df = FCT.FCMeasurement(ID=record[Names.FILENAME],
                                    datafile=os.path.join(data_dir, record[Names.FILENAME])).read_data()
        if max_records is not None:
            data_df = data_df[0:min(len(data_df), max_records)]
        elif fraction is not None:
            data_df = data_df.sample(frac=fraction, replace=True)
            #data_df = data_df.replace([np.inf, -np.inf], np.nan)
        #data_df = data_df[~data_df.isin(['NaN', 'NaT']).any(axis=1)]
        
        data_df[Names.FILENAME] = record[Names.FILENAME]
        all_data_df = all_data_df.append(data_df)

    ## Join data and metadata
    final_df = metadata_df.merge(all_data_df, left_on='filename', right_on='filename', how='outer')    
    return final_df

def get_xplan_data_and_metadata_df(metadata_df, data_dir, fraction=None, max_records=None):
    """
    Rename columns from data and metadata to match xplan columns
    """
    df = get_data_and_metadata_df(metadata_df, data_dir, fraction=fraction, max_records=max_records)
    rename_map = {
        "experiment_id" : "plan",
        "sample_id" : "id",
        "strain_input_state" : "input",
        "strain_circuit" : "gate",
        "strain_sbh_uri" : "strain"        
    }
    
    #pipeline columns
    #Index([ 'lab',  'output',
    #    
    #   'strain_sbh_uri', 'Time', 'FSC-A', 'SSC-A', 'BL1-A', 'RL1-A', 'FSC-H',
    #   'SSC-H', 'BL1-H', 'RL1-H', 'FSC-W', 'SSC-W', 'BL1-W', 'RL1-W'],
    #  dtype='object')
    
    #xplan columns
    #['Unnamed: 0',  'replicate',
    #   'od', 'bead', 'filename', 'gate', 'Time', 'FSC-A', 'SSC-A', 'BL1-A',
    #   'FSC-H', 'SSC-H', 'BL1-H', 'FSC-W', 'SSC-W', 'BL1-W']
    
    df = df.rename(index=str, columns=rename_map)
    return df
    
###############################################
# Helpers for getting sample data to classify #
###############################################

def gate_output(gate, inputs):
    if gate == Names.NOR:
        if inputs == Names.INPUT_00:
            return 1
        else:
            return 0
    elif gate == Names.AND:
        if inputs == Names.INPUT_11:
            return 1
        else:
            return 0
    elif gate == Names.NAND:
        if inputs == Names.INPUT_11:
            return 0
        else:
            return 1
    elif gate == Names.OR:
        if inputs == Names.INPUT_00:
            return 0
        else:
            return 1
    elif gate == Names.XOR:
        if inputs == Names.INPUT_00 or inputs == Names.INPUT_11:
            return 0
        else:
            return 1
    elif gate == Names.XNOR:
        if inputs == Names.INPUT_00 or inputs == Names.INPUT_11:
            return 1
        else:
            return 0

def get_strain_dataframe_for_classifier(circuit, input, od=0.0003, media='SC Media', experiment='', data_dir='', fraction=None):
    """
    """
    meta_df = get_metadata_dataframe(get_strain(circuit, input, od=od, media=media, experiment=experiment))
    da = meta_df[[Names.FILENAME, 'output']].copy()
    #da['strain'] = da['strain'].mask(da['strain'] == 'WT-Dead-Control',  0)
    #da['strain'] = da['strain'].mask(da['strain'] == 'WT-Live-Control',  1)
    da = get_data_and_metadata_df(da, data_dir, fraction=fraction)
    #da = da.drop(columns=['fcs_files', 'Time', 'RL1-W', 'BL1-W'])
    da = da.drop(columns=['filename', 'Time'])
    return da


def get_strain(strain_circuit, strain_input_state,od=0.0003, media='SC Media',experiment=''):
    query={}
    query[Names.CHALLENGE_PROBLEM] = Names.YEAST_STATES
    query[Names.FILE_TYPE] = Names.FCS
    query[Names.LAB] = Names.TRANSCRIPTIC
    
#    if strain_circuit == 'XOR' and strain_input_state == '00':
#        query['strain'] = '16970'
#    else:
    query[Names.STRAIN_CIRCUIT] = strain_circuit
    query[Names.STRAIN_INPUT_STATE] = strain_input_state
    query[Names.INOCULATION_DENSITY_VALUE] =  od
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
    query['challenge_problem'] = 'YEAST_STATES'
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

def get_experiment_ids(challenge_problem='YEAST_STATES', lab='Transcriptic'):
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
    