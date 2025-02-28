## Redefines function for pytorch and numpy backends, so that they have same behavior
from copy import deepcopy

import numpy as np
import torch

from pgmpy import config


def size(arr):
    if isinstance(arr, np.ndarray):
        return arr.size
    else:
        return arr.nelement()


def copy(arr):
    if config.get_backend() == "numpy":
        if isinstance(arr, np.ndarray):
            return np.array(arr)
        elif isinstance(arr, (int, float)):
            return deepcopy(arr)
    else:
        return torch.tensor(arr, dtype=config.get_dtype(), device=config.get_device())


def tobytes(arr):
    if isinstance(arr, np.ndarray):
        return arr.tobytes()
    else:
        return arr.numpy(force=True).tobytes()


def max(arr, axis=None):
    if axis is not None:
        axis = tuple(axis)

    if isinstance(arr, np.ndarray):
        return np.max(arr, axis=axis)
    else:
        return torch.amax(arr, dim=axis)


def einsum(*args):
    if config.get_backend() == "numpy":
        return np.einsum(*args)
    else:
        return torch.einsum(*args)


def argmax(arr):
    if isinstance(arr, np.ndarray):
        return np.argmax(arr)
    else:
        return torch.argmax(arr)


def stack(arr_iter):
    if config.get_backend() == "numpy":
        return np.stack(tuple(arr_iter))
    else:
        return torch.stack(tuple(arr_iter))


def to_numpy(arr, decimals=None):
    if isinstance(arr, torch.Tensor):
        if arr.device.type.startswith("cuda"):
            arr = arr.cpu().detach().numpy()
        else:
            arr = arr.numpy(force=True)

    if decimals is None:
        return np.array(arr)
    else:
        return np.array(arr).round(decimals)


def ravel_f(arr):
    if isinstance(arr, np.ndarray):
        return arr.ravel("F")
    else:
        return to_numpy(arr).ravel("F")


def ones(n):
    if config.get_backend() == "numpy":
        return np.ones(n, dtype=config.get_dtype())

    else:
        return torch.ones(n, dtype=config.get_dtype(), device=config.get_device())


def get_compute_backend():
    if config.get_backend() == "numpy":
        return np
    else:
        return torch


def unique(arr, axis=0, return_counts=False, return_inverse=False):
    if isinstance(arr, np.ndarray):
        return np.unique(
            arr, axis=axis, return_counts=return_counts, return_inverse=return_inverse
        )
    else:
        return torch.unique(
            arr, return_inverse=return_inverse, return_counts=return_counts, dim=axis
        )


def flip(arr, axis=0):
    if isinstance(arr, np.ndarray):
        return np.flip(arr, axis=axis)
    else:
        return torch.flip(arr, dims=axis)


def transpose(arr, axis):
    if isinstance(arr, np.ndarray):
        return np.transpose(arr, axes=axis)
    else:
        return torch.permute(arr, dims=axis)


def exp(arr):
    if isinstance(arr, np.ndarray):
        return np.exp(arr)
    else:
        return arr.exp()


def sum(arr):
    if isinstance(arr, np.ndarray):
        return np.sum(arr)
    else:
        return torch.sum(arr)


def allclose(arr1, arr2, atol):
    if isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
        return np.allclose(arr1, arr2, atol=atol)
    else:
        return torch.allclose(
            torch.tensor(arr1, dtype=config.get_dtype(), device=config.get_device()),
            torch.tensor(arr2, dtype=config.get_dtype(), device=config.get_device()),
            atol=atol,
        )
