from .struct import MatStruct
from .hdf5 import loadmat as hdf5_loadmat
from scipy.io import loadmat as scipy_loadmat
from pathlib import Path
from typing import Union
import numpy as np


def loadmat(files: Union[str, list]) -> MatStruct:
    """
    Load a MATLAB .mat file or group of files, supporting both HDF5 and legacy formats.
    This function attempts to load the file using the HDF5 format first,
    falling back to the legacy format (via scipy) if necessary.

    Parameters
    ----------
    fpath : str or list of str
        Path to the .mat file.

    Returns
    -------
    MatStruct
        A structured object containing the MATLAB variables, mimicking the MATLAB struct format.
    """

    data = {}
    if isinstance(files, str):
        files = [Path(files)]
    else:
        files = [Path(f) for f in files]

    for fpath in files:
        try:
            # Attempt to load the file using HDF5 format
            data[fpath.stem] = hdf5_loadmat(fpath)
        except Exception as e:
            if "file signature not found" in e.args[0]:
                # If the file is not in HDF5 format, fall back to scipy's loadmat
                data[fpath.stem] = scipy_loadmat(fpath, squeeze_me=True)
                # Remove metadata keys that are not needed in the structured array
                data[fpath.stem].pop("__header__")
                data[fpath.stem].pop("__version__")
                data[fpath.stem].pop("__globals__")
                # Convert dict to structured array
                dtype = [(k, "O") for k in data[fpath.stem].keys()]
                data[fpath.stem] = np.void(
                    tuple(data[fpath.stem].values()), dtype=dtype
                )
            else:
                # If the error is not related to file format, re-raise it
                raise e
    if len(files) == 1:
        # If only one file is provided, return a MatStruct with that data
        return MatStruct(data[files[0].stem], name=files[0].stem)
    else:
        # If multiple files are provided, return a MatStruct with all data
        return MatStruct(data)
