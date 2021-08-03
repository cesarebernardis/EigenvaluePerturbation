
def invert_dictionary(id_to_index):

    index_to_id = {}

    for id in id_to_index.keys():
        index = id_to_index[id]
        index_to_id[index] = id

    return index_to_id


def estimate_sparse_size(num_rows, topK):
    """
    :param num_rows: rows or colum of square matrix
    :param topK: number of elements for each row
    :return: size in Byte
    """

    num_cells = num_rows*topK
    sparse_size = 4*num_cells*2 + 8*num_cells

    return sparse_size


def seconds_to_biggest_unit(time_in_seconds, data_array=None):

    conversion_factor = [
        ("sec", 60),
        ("min", 60),
        ("hour", 24),
        ("day", 365),
    ]

    terminate = False
    unit_index = 0

    new_time_value = time_in_seconds
    new_time_unit = "sec"

    while not terminate:

        next_time = new_time_value/conversion_factor[unit_index][1]

        if next_time >= 1.0:
            new_time_value = next_time

            if data_array is not None:
                data_array /= conversion_factor[unit_index][1]

            unit_index += 1
            new_time_unit = conversion_factor[unit_index][0]

        else:
            terminate = True

    if data_array is not None:
        return new_time_value, new_time_unit, data_array

    else:
        return new_time_value, new_time_unit
