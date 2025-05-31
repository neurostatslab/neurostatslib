from .types import MatStruct
from .hdf5 import loadmat as hdf5_loadmat
from scipy.io import loadmat as scipy_loadmat
from pathlib import Path


def loadmat(filename):
    try:
        data = hdf5_loadmat(filename)
    except:
        data = scipy_loadmat(filename, squeeze_me=True)

    return MatStruct(data, name=Path(filename).stem)
