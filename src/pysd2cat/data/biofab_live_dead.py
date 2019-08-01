import pandas as pd
import os
import sys
import glob
import json
from os.path import expanduser
import matplotlib.pyplot as plt                   # For graphics
import numpy as np
from agavepy.agave import Agave, AgaveError
from agavepy.files.download import files_download
from pysd2cat.data import pipeline
from pysd2cat.analysis.Names import Names

def make_experiment_metadata_dataframe(experiment_id):
    df = pd.DataFrame()
    samples = pipeline.get_experiment_samples(experiment_id,file_type='FCS')
    for sample in samples:
        df = df.append(sample_to_row(sample), ignore_index=True)
    return df

def flatten_sample_contents(feature, value):
    kv_pairs=[]
    for content in value:
        if "name" in content:
            if 'label' in content['name']:
                if content['name']['label'] == 'YPAD':
                    kv_pairs.append(('media', content['name']['label']))
                elif content['name']['label'] == 'Ethanol':
                    kv_pairs.append(('kill_method', 'Ethanol'))
                    if 'volume' in content:
                        if 'unit' in content['volume'] and 'value' in content['volume']:
                            kv_pairs.append(('kill_volume', content['volume']['value']))
                            kv_pairs.append(('kill_volume_unit', content['volume']['unit']))
                        else:
                            raise "Malformed Volume of sample contents: " + str(content['volume'])
                elif content['name']['label'] == 'SYTOX Red Stain':
                    #print(content['name'])
                    kv_pairs.append(('stain', 'SYTOX Red Stain'))
                    if 'volume' in content:
                        if 'unit' in content['volume'] and 'value' in content['volume']:
                            kv_pairs.append(('stain_volume', content['volume']['value']))
                            kv_pairs.append(('stain_volume_unit', content['volume']['unit']))
                        else:
                            raise "Malformed Volume of sample contents: " + str(content['volume'])
                    
    return kv_pairs

def flatten_temperature(feature, value):
    return [(feature, value['value'])]

def flatten_feature(feature, value):
    if feature == 'sample_contents':
        return flatten_sample_contents(feature, value)
    elif feature == 'temperature':
        return flatten_temperature(feature, value)
    else:
        raise "Cannot flatten feature: " + feature

def sample_to_row(sample):
    features = [k for k,v in sample.items() if type(v) is not dict and type(v) is not list]
    row = {}
    for feature in features:
        row[feature] = sample[feature]
        
    #Handle nested features
    for feature in [x for x in sample.keys() if x  not in features]:
        kv_pairs = flatten_feature(feature, sample[feature])
        for k, v in kv_pairs:
            row[k] = v
    return row

def fetch_data(meta_df, data_dir, overwrite=False):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    ag = Agave.restore()

    for i, row in meta_df.iterrows():
        src = row['agave_system'] + row['agave_path']
        dest = data_dir + row['agave_path']

        if overwrite or not os.path.exists(dest):
            result_dir = "/".join((data_dir + row['agave_path']).split('/')[0:-1])
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            #print(src)
            print(dest)
            
            files_download(ag.api_server, ag.token.token_info['access_token'], src, dest)
    
def write_meta_and_data_dataframe(experiment, overwrite=False):
    data_dir = os.path.join('data/biofab', experiment)
    all_data_file = os.path.join(data_dir, 'data.csv')

    if overwrite or not os.path.exists(all_data_file):
        meta_df = make_experiment_metadata_dataframe(experiment)
        fetch_data(meta_df, data_dir)
        meta_df[Names.FILENAME] = meta_df.apply(lambda x:  data_dir + "/" + x['agave_path'], axis=1)
        all_data_df = pipeline.get_data_and_metadata_df(meta_df, '.', fraction=None, max_records=None)
        all_data_df.to_csv(all_data_file)

def get_experiment_data_df(experiment):
    data_dir = os.path.join('data/biofab', experiment)
    all_data_file = os.path.join(data_dir, 'data.csv')
    experiment_df = pd.read_csv(all_data_file, index_col=0)
    return experiment_df
