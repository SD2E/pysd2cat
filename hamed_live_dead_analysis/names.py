class Names:
    index = "arbitrary_index"
    time = "time_point"
    stain = "stain"
    label = "label"
    label_preds = "label_predictions"
    data_file_name = "pipeline_data.csv"

    # strains
    yeast = "yeast"
    basc = "basc"
    ecoli = "ecoli"

    # treatments
    ethanol = "ethanol"
    heat = "heat"
    treatments_dict = {
        ethanol: [0, 140, 210, 280, 1120],
        heat: [0]
    }
    timepoints = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    # feature columns
    morph_cols = ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W"]
    sytox_cols = ["RL1-A", "RL1-H", "RL1-W"]
    # mito_cols = None
    morph_cols = ["log_{}".format(x) for x in morph_cols]
    sytox_cols = ["log_{}".format(x) for x in sytox_cols]

    # experiment dictionary
    exp_dict = {
        (yeast, ethanol): "temporary_yeast_ethanol",
        (basc, ethanol): "experiment.transcriptic.r1eaf248xavu8a",
        (ecoli, ethanol): "experiment.transcriptic.r1eaf25ne8ajts"
    }
    exp_data_dir = "experiment_data"
    harness_output_dir = "test_harness_outputs"
    # each experiment should have a corresponding folder with the same name as the experiment_id
    # inside the folder you will have data files: dataset, train, test, normalized_train, normalized_test, etc.
    # then the LiveDeadPipeline can call file if it exists or otherwise create it using preprocessing methods

    num_live = "num_live"
    num_dead = "num_dead"
    percent_live = "predicted %live"
