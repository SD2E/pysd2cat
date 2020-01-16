"""
Converts container data received from strateos api into format expected
by the SAT problem generator.
"""

import string


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


def aliquot_dict(well_map, aliquots_df, well_idx):
    """
    Create a dictionary of aliquot info for the given well in the container.
    """
    # aliquot_info = aliquots_df.loc[well_idx]
    return {"strain": well_map[well_idx]}


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


def container_to_dict(container):
    """
    Convert a transcriptic container object into a dict format
    expected by SAT problem generator
    """
    col_count = container.attributes['container_type']['col_count']
    well_map = container.well_map
    return {
        "aliquots": {
            container_well_idx_name(col_count, well_idx): aliquot_dict(well_map, container.aliquots, well_idx)
            for well_idx in well_map.keys()
        },
        "columns": column_dict(col_count, well_map.keys())
    }
