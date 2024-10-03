import sys
from datetime import datetime, timedelta
from random import shuffle

import numpy as np

sys.path.insert(1, "./Data_processing")
from sql_util import create_connection, execute_read_query

# months for testing set
test_months = (11, 12)

def get_datetime(date_str):
    date = datetime.strptime(date_str, "%Y.%m.%d_%H:%M:%S")
    return date


# FN = filenames, NC_IN = #channels in input, NC_OUT = #channels in output
# This function essentially reads the image, and shifts it slightly by up
# to 15 pixels any direction before returning it. This is probably to
# prevent overfitting
def READ_IMAGE(FN, NC_IN, NC_OUT):
    IMG_A = np.nan_to_num(np.load(FN[0]))
    IMG_B = np.nan_to_num(np.load(FN[1]))
    X, Y = np.random.randint(31), np.random.randint(31)
    if NC_IN != 1:
        IMG_A = np.pad(IMG_A, ((15, 15), (15, 15), (0, 0)), "constant")
        IMG_A = IMG_A[X : X + 1024, Y : Y + 1024, :]
    else:
        IMG_A = np.pad(IMG_A, 15, "constant")
        IMG_A = IMG_A[X : X + 1024, Y : Y + 1024]

    if NC_OUT != 1:
        IMG_B = np.pad(IMG_B, ((15, 15), (15, 15), (0, 0)), "constant")
        IMG_B = IMG_B[X : X + 1024, Y : Y + 1024, :]
    else:
        IMG_B = np.pad(IMG_B, 15, "constant")
        IMG_B = IMG_B[X : X + 1024, Y : Y + 1024]

    return IMG_A, IMG_B


# create mini batches for training (actually creates a generator
# that generates each element of the batch)
def MINI_BATCH(DATA_AB, BATCH_SIZE, NC_IN, NC_OUT):
    LENGTH = len(DATA_AB)
    EPOCH = i = 0
    TMP_SIZE = None
    while True:
        SIZE = TMP_SIZE if TMP_SIZE else BATCH_SIZE
        # if we reach the end of the data (which corresponds to an
        # epoch), shuffle data and begin again
        if i + SIZE > LENGTH:
            shuffle(DATA_AB)
            i = 0
            EPOCH += 1
        DATA_A = []
        DATA_B = []
        # make batches of length: SIZE
        for j in range(i, i + SIZE):
            IMG_A, IMG_B = READ_IMAGE(DATA_AB[j], NC_IN, NC_OUT)
            DATA_A.append(IMG_A)
            DATA_B.append(IMG_B)
        DATA_A = np.float32(DATA_A)
        DATA_B = np.float32(DATA_B)
        i += SIZE
        TMP_SIZE = yield EPOCH, DATA_A, DATA_B


def GRAB_DATA(data_paths, tol):
    # tolerance on image time difference in hours
    tol = timedelta(hours=tol)
    n = len(data_paths)
    input_paths, output_paths, input_dates, output_dates = np.array(data_paths).T

    for i in range(n):
        input_path = input_paths[i]
        input_date = get_datetime(input_dates[i])
        output_path = output_paths[i]
        output_date = get_datetime(output_dates[i])

        # accidentally inserted the string "NULL" into db, so this is to catch that
        if "NULL" in (input_path, output_path):
            continue

        if None in (input_path, output_path):
            continue

        if abs(input_date - output_date) > tol:
            continue

        if input_date.month in test_months:
            continue

        yield (input_path, output_path)


def get_data_paths(db_path, sql_input, sql_output, sql_connector):
    sql_tables = [data.split(".")[0] for data in [sql_input, sql_output]]
    query = f"""
SELECT
    {sql_input},
    {sql_output},
    {sql_tables[0]}.date,
    {sql_tables[1]}.date
FROM
    {sql_tables[0]},
    {sql_tables[1]}
WHERE
    {sql_connector[0]}={sql_connector[1]}
GROUP BY
    {sql_connector[0]}
"""
    connection = create_connection(db_path)
    data_paths = execute_read_query(connection, query)
    return data_paths
