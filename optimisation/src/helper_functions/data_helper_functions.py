import numpy as np


def strided_app(a: np.ndarray, L: int, S: int) -> np.ndarray:
    """
    Extract rolling windows from a numpy array

    :param a:   Array to extract windows from
    :param L:   Length of window
    :param S:   Distance between each strided window
    :return:    Matrix of strides
    """
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))


def timedelta_to_fraction_of_year(timedelta):
    """
    Converts np.timedelta[ns] data into time as a fraction of a year
    :param timedelta:   Timedelta to convert
    :return:            Converted time as a fraction of a year
    """
    year_in_nanoseconds = np.timedelta64(365, 'D') / np.timedelta64(1, 'ns')  # Duration of a year in nanoseconds
    timedelta_ns = timedelta / np.timedelta64(1, 'ns')
    fraction_of_year = timedelta_ns / year_in_nanoseconds
    return fraction_of_year
