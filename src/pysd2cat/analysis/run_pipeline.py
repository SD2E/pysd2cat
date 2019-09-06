from pysd2cat.data import tx_od, tx_fcs
from pysd2cat.analysis.Names import Names
from pysd2cat.analysis.live_dead_analysis import add_live_dead, get_fcs_columns
from pysd2cat.data.pipeline import get_xplan_data_and_metadata_df
from pysd2cat.analysis.correctness import compute_correctness_all
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
        overwrite = True,
        output_id_strain = { "A12" : {"strain": "WT-Live-Control"},
                                  "B12" : {"strain": "WT-Dead-Control"},
                                  "C12"  : {"strain": "NOR-00-Control"}}
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
            if row['output_id'] in output_id_strain:
                metadata = metadata.append({ "strain" : output_id_strain[laliquot.upper()]['strain'],
                                             "output_id" : laliquot.upper() },
                                            ignore_index=True)
            
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

        if 'live' not in fcs_df.columns or len(fcs_df.live.dropna()) == 0:
            ## Add live column
            logger.info("Adding live column")
            logger.debug(fcs_df.columns)
            strains = fcs_df.strain.unique()
            logger.info("Strains are: " + str(strains))
            if Names.WT_DEAD_CONTROL in strains and Names.WT_LIVE_CONTROL in strains:
                try:
                    fcs_df = add_live_dead(fcs_df, Names.STRAIN, Names.WT_LIVE_CONTROL, Names.WT_DEAD_CONTROL, fcs_columns=get_fcs_columns())
                    
                    #logger.info("Adding live column...")
                    #robj.logger.info(df1.shape)
                    #logger.info(df1['live'])
                    # Write dataframe as csv
                    #logger.info("Writing: " + run_data_file_stash)
                    #df1.to_csv(run_data_file_stash)
                except Exception as e:
                    logger.debug("Problem with live dead classification: " + str(e))
        logger.debug(fcs_df)

        correctness_df = compute_correctness_all(fcs_df, out_dir=challenge_out_dir)

        logger.debug(correctness_df)
        
    get_od = False
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
        od_meta_and_data_df.loc[:, 'pre_well'] = od_meta_and_data_df.apply(lambda x: x['pre_well'].upper(), axis=1)
        od_meta_and_data_df.loc[:, 'post_well'] = od_meta_and_data_df.apply(lambda x: x['post_well'].upper(), axis=1)
        od_meta_and_data_df = od_meta_and_data_df.drop(columns=['SynBioHub URI']).rename(columns={'post_well' : 'output_id'})
        od_meta_and_data_df = od_meta_and_data_df.merge(metadata, on='output_id', how='outer')
#        od_meta_and_data_df['part_1_id'] = experiment['part_1_id']
#        od_meta_and_data_df['part_2_id'] = experiment['part_2_id']
#        od_meta_and_data_df['calibration_id'] = experiment['calibration_id']

        

        logger.debug(od_meta_and_data_df)
    

    ## Process the data

    ## Save the data
