from typing import Union, List, Tuple
import numpy as np


def AddTime(paths: np.ndarray) -> np.ndarray:
    """
    Adds a time channel to a bank of paths. Paths are an N x l x d tensor. Output is an N x l x d+1 tensor.
    Time is added as the first dimension. It is by default a linear interpolation between [0, 1] with l steps.

    :param paths:   Input paths to add time channel to
    :return:        Paths with time channel added
    """
    N, l, d = paths.shape
    res = np.zeros((N, l, d+1))

    time = np.linspace(0, 1, l)

    res[..., 0]  = time
    res[..., 1:] = paths

    return res


def TranslatePaths(paths: np.ndarray) -> np.ndarray:
    """
    Translates paths to start at 0.

    Note that the signature is invariant to translations. However, for scaling purposes, it is sometimes helpful
    to have paths start at 0 (instead of some scaler such as 1)

    :param paths:   Paths to be translated
    :return:        Translated paths
    """

    _, l, _ = paths.shape

    return paths - np.tile(np.expand_dims(paths[:, 0, :], 1), (1, l, 1))


def ScalePaths(paths: np.ndarray, lambda_ : Union[List[float], Tuple[float], float]) -> np.ndarray:
    """
    Scales paths by a given scaler or set of scalers.

    :param paths:       Paths to be scaled, N x l x d
    :param lambda_:     Scaling. Can be either a float (scaling applied uniformly to each channel), a list of size
                        (d-1), which (assuming time has been added) only scales the state space channels, or a list of
                        size d, which scales each channel according to the corresponding value in the list
    :return:            Scaled paths
    """
    N, l , d = paths.shape
    res = np.zeros((N, l, d))
    if type(lambda_) == float:
        res = paths*lambda_
        return res
    elif (isinstance(lambda_, list) or isinstance(lambda_, tuple)) and all(isinstance(element, float) for element in lambda_):
        if len(lambda_) == d-1:
            res[..., 0] = paths[..., 0].copy()
            f_ind = 1
        elif len(lambda_) == d:
            f_ind = 0
        else:
            print("Not enough scalings for list-type scaling object. Please supply more scalings.")
            raise AssertionError
        for i, lambd_ in enumerate(lambda_):
            res[..., f_ind + i] = lambd_*paths[..., f_ind + i].copy()
    else:
        print("Incorrectly supplied scaling object. Check the documentation for more details")
        raise AssertionError
    return res


def LeadLag(paths: np.ndarray) -> np.ndarray:
    """
    Performs the lead-lag transformation on a bank of N x l x d paths. Becomes N x 2l + 1 x 2d.

    :param paths:   Bank to have lead-lag applied to
    :return:        Lead-lagged paths
    """
    _r = np.repeat(paths, 2, axis=1)
    _cpath = np.concatenate([_r[:, :-2], _r[:, 2:, :]], axis=2)
    _start = np.expand_dims(np.c_[_r[:, 0], _r[:, 0]], 1)

    return np.concatenate([_start, _cpath], axis=1)


def HoffLeadLag(paths: np.ndarray) -> np.ndarray:
    """
    Performs Hoff lead-lag transformation on a bank of paths. Sends N x l x d to N x 4l + 1 x 2d.

    :param paths:   Bank to have Hoff lead-lag applied to
    :return:        Hoff lead-lag transformed paths
    """
    _r = np.repeat(paths, 4, axis=1)
    _cpath = np.concatenate([_r[:, :-5], _r[:, 5:]], axis=2)
    _start = np.expand_dims(np.c_[_r[:, 0], _r[:, 0]], 1)

    return np.concatenate([_start, _cpath], axis=1)
