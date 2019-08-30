from pysd2cat.data import tx_od, tx_fcs
from pysd2cat.data.pipeline import get_xplan_data_and_metadata_df
import pandas as pd
import logging
import os

l = logging.getLogger(__file__)
l.setLevel(logging.DEBUG)


def run(
        metadata,
        experiment_id,
        batch_id,
        run_id_part_1,
        run_id_part_2,
        tx_email,
        tx_token,
        challenge_out_dir,        
        logger=l,
        local=False,
        overwrite = True
):
    """
    Retreive the data from the experiment and run it through processing pipeline.
    Save files: 
    1) FCS data with live/dead, high/low predictions
    2) Experiment Design 
    3) OD data
    4) Update master summary of all experiments
    5) 
    """

    ## Get all the data
    get_fcs = True
    if get_fcs:
    
        # Download all fcs files into $PWD/fcs/  Will be a number of *.fcs files there to upload
        fcs_files = tx_fcs.create_fcs_manifest_and_get_files(
            run_id_part_2,
            tx_email,
            tx_token,
            fcs_path=challenge_out_dir,
            download_zip=False,
            logger=logger,
            source_container_run=run_id_part_1)
        fcs_measurements = pd.DataFrame()
        for laliquot, record in fcs_files['aliquots'].items():
            logger.debug(record)
            mfile = record['file']
            row = {"output_id" : laliquot.upper(), "filename" : mfile}
            if 'control' in record['properties']:
                #row['strain'] = record['properties']['control']
                metadata = metadata.append({ "strain" : row['strain'], "output_id" : laliquot.upper() }, ignore_index=True)
            fcs_measurements = fcs_measurements.append(row, ignore_index=True)

        logger.debug(fcs_measurements)


        if local:
            max_records = 10
        else:
            max_records = 30000

        logger.info("Creating dataframe from experiment data ...")
        # Get all data in one dataframe
        data_dir = os.path.join(challenge_out_dir, 'fcs')
        fcs_df = get_xplan_data_and_metadata_df(fcs_measurements, data_dir, max_records=max_records)
        fcs_df = fcs_df.drop(columns=['filename'])
        fcs_df = fcs_df.merge(metadata, on='output_id', how='outer')
        #fcs_df.loc[:,'strain'] = fcs_df.apply(lambda x: x['strain_y'] if x['strain_x'] is None else x['strain_x'], axis=1)
        #logger.debug(fcs_df)
        #logger.debug(fcs_df.strain.unique())
        fcs_df.to_csv('dan.csv')

        if 'live' not in fcs_df.columns or len(fcs_df.live.dropna()) == 0:
            ## Add live column
            logger.info("Adding live column")
            
            strains = fcs_df.strain.unique()
            logger.info("Strains are: " + str(strains))
            if Names.WT_DEAD_CONTROL in strains and Names.WT_LIVE_CONTROL in strains:
                try:
                    logger.info(df.shape)
                    df1 = live_dead_analysis.add_live_dead(fcs_df, Names.STRAIN_NAME, Names.WT_LIVE_CONTROL, Names.WT_DEAD_CONTROL, fcs_columns=live_dead_analysis.get_fcs_columns())
                    logger.info("Adding live column...")
                    #robj.logger.info(df1.shape)
                    logger.info(df1['live'])
                    # Write dataframe as csv
                    logger.info("Writing: " + run_data_file_stash)
                    #df1.to_csv(run_data_file_stash)
                except Exception as e:
                    robj.logger.debug("Problem with live dead classification: " + str(e))

        
    get_od = True
    if get_od:
        # Get OD data
        ## TODO get calibration_id
        experiment = {
            "part_1_id" : run_id_part_1,
            "part_2_id" : run_id_part_2,
            "calibration_id" : None
            }
        od_out_dir = os.path.join(challenge_out_dir, 'data/transcriptic/od_corrected')
        if not os.path.exists(od_out_dir):
            os.makedirs(od_out_dir)

        od_meta_and_data_df = tx_od.get_experiment_data(experiment, od_out_dir, overwrite=overwrite)
        od_meta_and_data_df['part_1_id'] = experiment['part_1_id']
        od_meta_and_data_df['part_2_id'] = experiment['part_2_id']
        od_meta_and_data_df['calibration_id'] = experiment['calibration_id']

        logger.debug(od_meta_and_data_df)
    

    ## Process the data

    ## Save the data
