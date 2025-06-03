from .struct import MatStruct
from .hdf5 import loadmat as hdf5_loadmat
from scipy.io import loadmat as scipy_loadmat
from pathlib import Path


def loadmat(fpath: str) -> MatStruct:
    """
    Load a MATLAB .mat file, supporting both HDF5 and legacy formats.
    This function attempts to load the file using the HDF5 format first,
    falling back to the legacy format (via scipy) if necessary.

    Parameters
    ----------
    fpath : str
        Path to the .mat file.

    Returns
    -------
    MatStruct
        A structured object containing the MATLAB variables, mimicking the MATLAB struct format.
    """
    try:
        # Attempt to load the file using HDF5 format
        data = hdf5_loadmat(fpath)
    except Exception as e:
        if "file signature not found" in e.args[0]:
            # If the file is not in HDF5 format, fall back to scipy's loadmat
            data = scipy_loadmat(fpath, squeeze_me=True)
        else:
            # If the error is not related to file format, re-raise it
            raise e

    return MatStruct(data, name=Path(fpath).stem)
