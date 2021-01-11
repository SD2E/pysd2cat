import os
import matplotlib
import pandas as pd
from pathlib import Path
from pysd2cat.analysis.live_dead_pipeline.names import Names as n
from pysd2cat.analysis.live_dead_pipeline.ld_pipeline_classes import LiveDeadPipeline, ComparePipelines

matplotlib.use("tkagg")
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', None)


# # in 2019 data, add noise to the 15 and 80 ethanol concentrations so they don't overlap
# cfu_data_2019.loc[cfu_data_2019["treatment_concentration"] == 15, "percent_live"] += 1
# cfu_data_2019.loc[cfu_data_2019["treatment_concentration"] == 80, "percent_live"] -= 1

def main():
    strain = n.yeast

    # non-stain model
    ldp_no_stain = LiveDeadPipeline(x_strain=strain, x_treatment=n.ethanol, x_stain=0,
                                    y_strain=None, y_treatment=None, y_stain=None)
    ldp_no_stain.load_data()
    ldp_no_stain.condition_method(
        live_conditions=None,
        dead_conditions=[
            # {n.inducer_concentration: 80.0, n.timepoint: 1.0},
            # {n.inducer_concentration: 80.0, n.timepoint: 2.0},
            # {n.inducer_concentration: 80.0, n.timepoint: 3.0},
            # {n.inducer_concentration: 80.0, n.timepoint: 4.0},
            # {n.inducer_concentration: 80.0, n.timepoint: 5.0},
            {n.inducer_concentration: 80.0, n.timepoint: 6.0},
            # {n.inducer_concentration: 20.0, n.timepoint: 1.0},
            # {n.inducer_concentration: 20.0, n.timepoint: 2.0},
            # {n.inducer_concentration: 20.0, n.timepoint: 3.0},
            # {n.inducer_concentration: 20.0, n.timepoint: 4.0},
            # {n.inducer_concentration: 20.0, n.timepoint: 5.0},
            {n.inducer_concentration: 20.0, n.timepoint: 6.0},
            # {n.inducer_concentration: 15.0, n.timepoint: 6.0},
            # {n.inducer_concentration: 12.5, n.timepoint: 6.0},
            # {n.inducer_concentration: 10.0, n.timepoint: 6.0}
        ])

    # print(ldp_no_stain.labeled_data_dict[n.condition_method])

    # ldp_no_stain.plot_percent_live_over_conditions(n.condition_method, False)
    # print()

    ldp_no_stain.boost_labels_via_neural_network(method=n.condition_method)
    # ldp_no_stain.plot_percent_live_over_conditions(n.condition_method, True)


if __name__ == '__main__':
    main()
