"""
Converts container data received from strateos api into format expected
by the SAT problem generator.
"""

import string
import math

class ColumnNotExistError(Exception):
    pass


def well_column(col_count, well_idx):
    """
    Zero-based column number of the given zero-based well index
    for a container with the given number of columns
    """
    return well_idx % col_count


def container_well_idx_name(col_count, well_idx):
    """
    Convert a container well index integer to a string name


    examples for 96 well plate:
    12, 0 -> a1
    12, 1 -> a2
    12, 12 -> b1

    Letter is the row. Number is the column.
    """
    row_idx = well_idx // col_count
    row = string.ascii_lowercase[row_idx]
    col = well_column(col_count, well_idx) + 1

    return "{}{}".format(row, col)


def aliquot_dict(well_map, aliquots_df, well_idx, strain_name="Name"):
    """
    Create a dictionary of aliquot info for the given well in the container.
    """
    # aliquot_info = aliquots_df.loc[well_idx]
    return {"strain": aliquots_df.at[well_idx,strain_name]}


def column_dict(col_count, well_idxs):
    """
    Create a dictionary of the columns in the container
    """
    result = {col: [] for col in range(col_count)}

    for well_idx in well_idxs:
        col = well_column(col_count, well_idx)
        result[col].append(container_well_idx_name(col_count, well_idx))

    return {
        "col{}".format(col + 1): wells
        for col, wells in result.items()
        if len(wells) > 0
    }


def drop_nan_strain_aliquots(c2d):
    """
    Remove aliquots whose strain is nan
    """
    aliquots = {}
    dropped_aliquots = []
    for aliquot_id, aliquot in c2d['aliquots'].items():
        #print(str(type(aliquot['strain'])) + " " + str(aliquot['strain']))
        if type(aliquot['strain']) is float and math.isnan(aliquot['strain']):
            dropped_aliquots.append(aliquot_id)            
        else:
            aliquots[aliquot_id] = aliquot
    columns = {}
    for col_id, col in c2d['columns'].items():
        col_aliquots = []
        for aliquot in col:
            if not aliquot in dropped_aliquots:
                col_aliquots.append(aliquot)
        if len(col_aliquots) > 0:
            columns[col_id] = col_aliquots
    return {
        "aliquots" : aliquots,
        "columns" : columns
        }

def container_to_dict(container, strain_name="Name", drop_nan_strain=True, convert_none_strain_to_mediacontrol=True):
    """
    Convert a transcriptic container object into a dict format
    expected by SAT problem generator
    """
    col_count = container.attributes['container_type']['col_count']
    well_map = container.well_map
    c2d = {
        "aliquots": {
            container_well_idx_name(col_count, well_idx): aliquot_dict(well_map, container.aliquots, well_idx, strain_name=strain_name)
            for well_idx in well_map.keys()
        },
        "columns": column_dict(col_count, well_map.keys())
    }

    if drop_nan_strain:
        c2d = drop_nan_strain_aliquots(c2d)
    if convert_none_strain_to_mediacontrol:
        for aliquot_id, aliquot in c2d['aliquots'].items():
            if 'strain' in aliquot and not aliquot['strain']:
                aliquot['strain'] = "MediaControl"
    return c2d

def generate_container(num_aliquots, strain_name="Name", dimensions=(8, 12)):
    well_map = { i : {} for i in range(0, num_aliquots)  }
    col_count = dimensions[1]
    return {
        "aliquots" : {
            container_well_idx_name(col_count, well_idx): {}
            for well_idx in well_map.keys()
        },
        "columns" : column_dict(col_count, well_map.keys())
        }
