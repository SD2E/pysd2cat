from pysd2cat.data import pipeline
import pandas as pd
import os
import glob
from os.path import expanduser
import transcriptic
import csv
import logging
    
############################################################################
## ETL related data functions available from pipeline
############################################################################

def get_project_run_groups(project_id):
    """
    Given a project_id, return a dataframe that has a row for each experiment.
    Each row provides the run id of part 1, part 2, and calibration. 
    """
    
    project = transcriptic.project(project_id)
    project_runs = project.runs()
    
    ## Get calibration run
    calibration_runs = project_runs[project_runs['Name'].str.contains('Plate Reader Calibration')]
    if len(calibration_runs) > 0:
        calibration_run = calibration_runs['id'].unique()[0]
    else:
        calibration_run = None

    part_one_runs = project_runs[project_runs['Name'].str.contains('YeastGatesPartI_')]
    part_one_runs.loc[:,'index'] = part_one_runs.apply(lambda x: x['Name'].split('_')[-1], axis=1)
    part_one_runs = part_one_runs.rename(columns={'id' : 'part_1_id'}).drop(columns=['Name'])
    #print(part_one_runs['part_1_id'])

    part_two_runs = project_runs[project_runs['Name'].str.contains('YeastGatesPartII')]
    part_two_runs.loc[:,'index'] = part_two_runs.apply(lambda x: x['Name'].split('_')[-1], axis=1)
    part_two_runs = part_two_runs.rename(columns={'id' : 'part_2_id'}).drop(columns=['Name'])
    #print(part_two_runs['part_2_id'])
    
    both_parts = part_one_runs.merge(part_two_runs, on="index").drop(columns=['index'])
    both_parts.loc[:, 'calibration_id'] = calibration_run
    both_parts.loc[:, 'project_id'] = project_id
    return both_parts

def get_run_groups(projects, project_runs_file):
    """
    Cache all run groups in projects into project_runs_file.  Before cache exists,
    this call can be costly due to the necessary data retrieval.
    """
    if os.path.exists(project_runs_file):
        df = pd.read_csv(project_runs_file, index_col=0)
    else:
        df = pd.DataFrame()
                
    for project_id in projects:
        project_id = project_id.split('/')[-1]
        if project_id not in df['project_id'].unique():
            df = df.append(get_project_run_groups(project_id), ignore_index = True)
    df.to_csv(project_runs_file)
    return df



def well_idx_to_well_id(idx):
    """
    Convert an well index in a 96-well plate (8x12) to [char][int] ID.
    0 -> a1
    ...
    95 -> h12
    """
    assert idx >= 0 and idx < (12 * 8), "Index not in 96-well plate"
    # a-h
    chars = [chr(ord('a') + x) for x in range(ord('h') - ord('a') + 1)]
    row, col = divmod(idx, 12)
    return '{}{}'.format(chars[row], col + 1)

def create_od_csv(run_id, path):
    """
    Write a CSV file at path for OD data associated with run_id. 
    The OD data is uncorrected and adds some minimal amount of metadata
    needed to merge it with more verbose metadata.
    """
    run_obj = get_tx_run(run_id)
    d = run_obj.data
        
    is_calibration = False    
    measurements = {}

    # Ensure the absorbance dataset always comes before fluorescence
    # in the iteration. Logic below depends on this.
    datasets = sorted(d['Datasets'], key=lambda x: x.operation)
    for dataset in d['Datasets']:
        is_calibration = 'container' in dataset.attributes and 'CalibrationPlate' in dataset.attributes['container']['label']
        if dataset.data_type == "platereader" and dataset.operation == "absorbance":

            # Collect OD data
            for well, values in dataset.raw_data.items():
                if well in measurements:
                    measurements[well]['od'] =  values[0]
                else:
                    measurements[well] = {'od': values[0]}

            # Collect sample uris for each well
            for alq in dataset.attributes['container']['aliquots']:
                well = well_idx_to_well_id(alq['well_idx'])
                if well not in measurements:
                    continue
                try:
                    if is_calibration:
                        measurements[well]['sample'] = alq['name'] + "_" + str(alq['volume_ul'])
                    else:
                        #print(alq)
                        if 'control' in alq['properties']:
                            measurements[well]['sample'] = alq['properties']['control']
                        elif 'Sample_ID' in alq['properties']:
                            measurements[well]['sample'] = alq['properties']['Sample_ID']
                        if 'SynBioHub URI' in alq['properties']:
                            measurements[well]['SynBioHub URI'] = alq['properties']['SynBioHub URI']
                        else:
                            measurements[well]['SynBioHub URI'] = None
                except KeyError:
                    raise KeyError('Found sample for well which did not have OD measurement')

        elif dataset.data_type == "platereader" and dataset.operation == "fluorescence":
            for well, values in dataset.raw_data.items():
                if well in measurements:
                    measurements[well]['gfp'] =  values[0]
                else:
                    measurements[well] = {'gfp': values[0]}
                    
            for alq in dataset.attributes['container']['aliquots']:
                well = well_idx_to_well_id(alq['well_idx'])
                if well not in measurements:
                    continue
                try:
                    if is_calibration:
                        measurements[well]['sample'] = alq['name'] + "_" + str(alq['volume_ul'])
                    else:
                        if 'control' in alq['properties']:
                            measurements[well]['sample'] = alq['properties']['control']
                        elif 'Sample_ID' in alq['properties']:
                            measurements[well]['sample'] = alq['properties']['Sample_ID']
                        if 'SynBioHub URI' in alq['properties']:
                            measurements[well]['SynBioHub URI'] = alq['properties']['SynBioHub URI']        
                        else:
                            measurements[well]['SynBioHub URI'] = None
                except KeyError:
                    raise KeyError('Found sample for well which did not have Fluorescence measurement')




    
    with open(path, 'w+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['well', 'od', 'gfp', 'sample', 'SynBioHub URI'])

        writer.writeheader()
        writer.writerows((_measurement_to_csv_dict(well, data)
                          for well, data in sorted(measurements.items())))


def get_tx_run(run_id):
    """
    Create a TX run object from run id.
    """
    
    try:
        # In the reactor, the connection will already have been
        # made at this point. Don't worry if this fails. If the
        # connection failed up above, the Run instantion will
        # still cause an exception.
        connect()
    except Exception:
        pass
    return objects.Run(run_id)

def get_tx_project(project_id):
    """
    Create a TX project object from project id.
    """
    
    try:
        # In the reactor, the connection will already have been
        # made at this point. Don't worry if this fails. If the
        # connection failed up above, the Run instantion will
        # still cause an exception.
        connect()
    except Exception:
        pass
    return objects.Project(project_id)



def _measurement_to_csv_dict(well, data):
    """
    Helper function to convert measurement created in create_od_csv to a csv row
    """
    res = data.copy()
    res['well'] = well
    return res

def get_od_scaling_factor(calibration_df):
    """
    Compute scaling factor from controls.
    """
    
    ludox_df = calibration_df.loc[calibration_df['sample'] == 'ludox_cal_100.0']
    water_df = calibration_df.loc[calibration_df['sample'] == 'water_cal_100.0']
    
    ludox_mean = ludox_df['od'].mean()
    water_mean = water_df['od'].mean()
    
    corrected_abs600 = ludox_mean - water_mean
    reference_od600 = 0.063
    scaling_factor = reference_od600 / corrected_abs600

    return scaling_factor


def make_experiment_corrected_df(calibration_df, part_1_df, part_2_df):
    """
    Join the run data to get pre and post ODs, then add corrections of 
    said ODs.
    """
    
    part_1_df = part_1_df.rename(columns={"od" : "pre_od_raw", "gfp" : "pre_gfp_raw", "well" : "pre_well"})
    part_2_df = part_2_df.rename(columns={"od" : "post_od_raw", "gfp" : "post_gfp_raw", "well" : "post_well"})
    #print(part_1_df['sample'].value_counts())
    #print(part_2_df['sample'].value_counts())
    experiment_df = part_1_df.merge(part_2_df, how='inner', on=['sample', 'SynBioHub URI']).drop(columns=['sample']).dropna().reset_index(drop=True)
    od_scaling_factor = get_od_scaling_factor(calibration_df)
    
    #print(experiment_df)
    
    experiment_df.loc[:, 'pre_od_corrected'] = experiment_df.apply(lambda x: x['pre_od_raw'] * od_scaling_factor, axis=1 )
    experiment_df.loc[:, 'post_od_corrected'] = experiment_df.apply(lambda x: x['post_od_raw'] * od_scaling_factor, axis=1 )
    
    return experiment_df

def get_experiment_data(experiment, out_dir):
    """
    For a triple of runs (part 1, part 2, and calibration), generate a
    dataframe with OD data for all wells.
    """
    
    #print(experiment)
    if not pd.isna(experiment['calibration_id']):
        calibration_file = os.path.join(out_dir, experiment['calibration_id'] + '.csv')
        part_1_file = os.path.join(out_dir, experiment['part_1_id'] + '.csv')
        part_2_file = os.path.join(out_dir, experiment['part_2_id'] + '.csv')
        
        if not os.path.exists(calibration_file):
            create_od_csv(experiment['calibration_id'], csv_path=calibration_file)
        if not os.path.exists(part_1_file):
            create_od_csv(experiment['part_1_id'], csv_path=part_1_file)
        if not os.path.exists(part_2_file):
            create_od_csv(experiment['part_2_id'], csv_path=part_2_file)
        
        calibration_df = pd.read_csv(calibration_file, index_col = False)
        part_1_df = pd.read_csv(part_1_file, index_col = False)
        part_2_df = pd.read_csv(part_2_file, index_col = False)
        
        experiment_df = make_experiment_corrected_df(calibration_df, part_1_df, part_2_df)
        return experiment_df
    else:
        raise Exception("Do not have calibration run for " + str(experiment))
   

def get_meta(experiment):
    """
    Get metadata for an experiment from the xplan-reactor state.
    """

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('main')

    xplan_state_file = os.path.join(expanduser("~"), 'tacc-work/xplan-reactor/state.json')
    state = json.load(open(xplan_state_file, 'r'))

    part_1_id = experiment['part_1_id'] #'r1c5vaeb8vbt9'
    #part_2_id = experiment['part_2_id'] #'r1c66mfpj7guh'

    xplan_experiment = state['runs'][part_1_id]
    #print(experiment)
    request_file = os.path.join(expanduser("~"), 'tacc-work/xplan-reactor/experiments/', xplan_experiment['experiment_id'], 'request_' + xplan_experiment['experiment_id'] + ".json")
    request = experiment_request.ExperimentRequest(**json.load(open(request_file, 'r')))
    meta = request.pd_metadata(logger, plates=[xplan_experiment['plate_id']])
    meta = meta.rename(columns={'gate' : 'strain_circuit'})

    return meta

def get_data_and_metadata_df(experiment, out_dir):
    """
    Get all metadata and data for an experiment (run triple),
    cache a copy in out_dir, and return it.
    """
    
    experiment_od_file = os.path.join(out_dir, experiment['part_1_id'] + "_" + experiment['part_2_id'] + '.csv')
    if not os.path.exists(experiment_od_file):
        try:
            data = get_experiment_data(experiment, out_dir) 
            meta = get_meta(experiment)
            meta_and_data_df = meta.merge(data,  left_on='well', right_on='post_well')
            meta_and_data_df.to_csv(experiment_od_file)
        except Exception as e:
            print("Could not generate dataframe for experiment: " + str(experiment) + " because " + str(e))
            return pd.DataFrame()
    else: 
        meta_and_data_df = pd.read_csv(experiment_od_file)

    return meta_and_data_df

#######

def get_etl_od_data():
    """
    Return dictionary mapping experiment id to data file
    """
    data_loc = 'sd2e-community/shared-q1-workshop/gzynda/platereactor_out/transcriptic/201808/yeast_gates'
    experiments = glob.glob(os.path.join(expanduser("~"), data_loc, '*'))
    results = {}
    for experiment in experiments:
        files = glob.glob(os.path.join(experiment, 'pyPlateCalibrate*', '*scaled_data.csv'))        
        ex_id = 'experiment.transcriptic.' + experiment.split('/')[-1]
        results[ex_id] = files
    return results

def get_meta_and_etl_od_data(experiment_id, od_files):
    """
    Create joint dataframe with metadata and etl'd od data
    """
    samples = pipeline.get_experiment_samples(experiment_id,file_type='CSV')
    calibration_samples = [x for x in samples if 'calibration' in x['filename']]
    ## Drop calibration_samples for now
    samples = [x for x in samples if x not in calibration_samples]
    
    assert(len(samples) > 0)
    
    #print(calibration_samples)
    #print("Got " + str(len(samples)) + " samples for " + experiment_id)

    meta = pipeline.get_metadata_dataframe(samples)
    #od_files = meta.filename.unique()
    #print(od_files)
    #od_dfs = {}
    #samples = {}
    #for od_file in od_files:
    #    od_dfs[od_file] = pd.read_csv(od_file)
        #samples[od_file] = od_dfs[od_file]['Sample_ID'].unique()
        
    dfs = join_meta_and_data(meta, od_files)
    
    return dfs

def join_meta_and_data(metadata_df, od_files):
    """
    Join based upon the "Sample_ID" in od_dfs and lab sample id in metadata_df
    """
    od_dfs = {}
    for od_df_file in od_files:
        od_df = pd.read_csv(od_df_file)
        od_df['Sample_ID'] = od_df['Sample_ID'].apply(lambda x: 'sample.transcriptic.' + x)
        od_df = od_df.merge(metadata_df, left_on='Sample_ID', right_on='sample_id', how='left') 
        od_dfs[od_df_file] = od_df

    return od_dfs

def get_all_sample_corrected_od_dataframe(od_files):
    df = pd.DataFrame()
    for out_file in od_files:
        if not os.path.isfile(out_file):
            continue
    
        plate_df = pd.read_csv(out_file, index_col=0)
        df = df.append(plate_df)
    df = df.loc[(df['strain'].notna())]
    return df

def merge_pre_post_od_dfs(first, second):
    """
    First has pre-dilution od, and second has final od
    """
    #print(first.columns)
    #print(second.columns)
    drop_cols = ['Sample', 'od', 'experiment_id', 'filename', 'lab', 'media',  'output','strain_circuit', 'strain_input_state', 'temperature']
    rename_map = {}
    for c in first.columns:
        if c not in ['experiment_id', 'filename', 'lab', 'media', 'od', 'output', 'replicate', 'strain', 'strain_circuit', 'strain_input_state', 'temperature']:
            rename_map[c] = 'pre_' + c
    first = first.drop(columns=drop_cols).rename(index=str, columns=rename_map)
    
    rename_map = {}
    for c in first.columns:
        if c not in ['experiment_id', 'filename', 'lab', 'media', 'od', 'output', 'replicate',  'strain', 'strain_circuit', 'strain_input_state', 'temperature']:
            rename_map[c] = 'post_' + c
    second = second.rename(index=str, columns=rename_map)
    
    return second.merge(first, on=['replicate', 'strain'], how='inner')

def make_etl_od_datafiles(data, out_dir):
    """
    Create one datafile per container that has corrected OD values.  Join
    datafiles for related containers into one for an experiment.
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for ex_id, od_files in data.items():
        for od_file in od_files:
            container = od_file.split('/')[-1].split('_')[0]
            out_file = os.path.join(out_dir, ex_id + '_' + str(container) + '.csv')
            if os.path.isfile(out_file):
                continue

            try:
                dfs = get_meta_and_etl_od_data(ex_id, [od_file])
                for k, v in dfs.items():
                    v.to_csv(out_file)
            except Exception as e:
                print("Could not process " + str(ex_id) + " because: " + str(e) )
                
    for ex_id, od_files in data.items():
        od_dfs = []
        out_file = os.path.join(out_dir, ex_id + '.csv')
        if os.path.isfile(out_file):
            continue
        if len(od_files) < 2:
            continue


    #    print(ex_id)
    #    print(od_files)
    #    assert(len(od_files) == 2)
        for od_file in od_files:
            container = od_file.split('/')[-1].split('_')[0]
            out_file = os.path.join(out_dir, ex_id + '_' + str(container) + '.csv')
            df = pd.read_csv(out_file, index_col=0)
            od_dfs.append(df)
        if len(od_dfs[0]) < len(od_dfs[1]):
            first = od_dfs[0]
            second = od_dfs[1]
        else:
            first = od_dfs[1]
            second = od_dfs[0]

        df = merge_pre_post_od_dfs(first, second)

        df.to_csv(out_file)


############################################################################
## Raw data functions available from pipeline
############################################################################

def make_raw_od_datafiles(ex_ids, out_dir):
    """
    Create one datafile per experiment that has uncorrected OD values.  
    """
    for ex_id in ex_ids:
        out_file = os.path.join(out_dir, ex_id + '.csv')
        if os.path.isfile(out_file):
            continue

        try:
            df = pipeline_od.get_experiment_od_data(ex_id)
            df.to_csv(out_file)
        except Exception as e:
            print("Could not process " + str(ex_id) + " because: " + str(e) )




def get_od_training_df(df):
    """
    Convert df into a dataframe suitable for regression.
    """
    df = df.rename(index=str, columns={'od_2': "output_od", 'od' : 'input_od'})
    #df = df[['od', 'media', 'temperature', 'strain', 'class_label']]
    df = df[['input_od', 'strain', 'output_od']]
    df = df.dropna(subset=['strain'])
    return df[['strain', 'output_od']], df['input_od']

def get_all_sample_od_dataframe(od_files):
    df = pd.DataFrame()
    for out_file in od_files:
        if not os.path.isfile(out_file):
            continue
    
        plate_df = pd.read_csv(out_file)
        df = df.append(plate_df)
    df = df.loc[(df['od_2'].notna()) & (df['strain'].notna())]
    return df

def get_experiment_od_and_metadata(metadata_df, od_dfs):
    #Strip leading sample id if applicable
    sample_ids = metadata_df.sample_id.unique()
    if len(sample_ids) > 0 and 'https' in sample_ids[0]:
        metadata_df['od_sample_id'] = metadata_df['sample_id'].apply(lambda x: ".".join(x.split('.')[2:]))
    elif len(sample_ids) > 0 and 'sample.transcriptic' in sample_ids[0]: 
        metadata_df['od_sample_id'] = metadata_df['sample_id'].apply(lambda x: x.split('.')[2])        
    else:
        metadata_df['od_sample_id'] = metadata_df['sample_id']

    #get collection id from od_dfs keys
    def get_collection_id_from_filename(filename):
        #print(filename)
        if 'od.csv' in filename:
            ## Assume filename has collection id in path
            collection_id = filename.split('/')[8]
            return collection_id
        else:
            ## Assume filename is of form od1.csv
            collection_id = filename.split('/')[-1].split('.')[0][2:]
            return collection_id
    
    collection_ids = {}
    for k,v in od_dfs.items():
        #print(k)
        collection_ids[get_collection_id_from_filename(k)] = k
    
    for k,v in collection_ids.items():
        #rename od_dfs columns with collection id index
        try:
            od_df = od_dfs[v].drop(columns=['container_id', 'aliquot_id'])
        except Exception as e:
            od_df = od_dfs[v]
        
        od_df.columns = [x + "_" + str(k) for x in od_df.columns]
        #print(metadata_df.shape)
        #print(od_df.columns)
        #print(metadata_df.columns)
        #print(k)
        
        metadata_df = metadata_df.merge(od_df, left_on='od_sample_id', right_on='sample'+ "_" + str(k), how='left') 
        metadata_df = metadata_df.drop(columns=["well_" + str(k), "sample_" + str(k)])
        #print(metadata_df.shape)
    metadata_df = metadata_df.drop(columns=['od_sample_id'])
    return metadata_df

def match_pre_post_od(df):
    """
    The dataframe has different sample ids for pre and post dilution samples.  The post
    samples are derived from pre samples, and this function adds the pre od for each sample 
    as a column for the corresponding post samples.
    """
    return df

def get_calibration_df(calibration_file):
    return pd.read_csv(calibration_file)






def get_strain_growth_plot(experiment_id):
    samples = pipeline.get_experiment_samples(experiment_id,file_type='CSV')
    calibration_samples = [x for x in samples if 'calibration' in x['filename']]
    ## Drop calibration_samples for now
    samples = [x for x in samples if x not in calibration_samples]
    
    #print(calibration_samples)
    #print("Got " + str(len(samples)) + " samples for " + experiment_id)
    
    
    
    meta = pipeline.get_metadata_dataframe(samples)
    od_files = meta.filename.unique()
    #print(od_files)
    od_dfs = {}
    for od_file in od_files:
        od_dfs[od_file] = pd.read_csv(od_file)
    df = get_experiment_od_and_metadata(meta, od_dfs)
    df = match_pre_post_od(df)
    return df

