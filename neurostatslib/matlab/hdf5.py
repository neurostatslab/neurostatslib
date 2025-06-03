import h5py
from scipy.sparse import csc_matrix
import numpy as np
from typing import Union

# class MatFileReader:


def get_variable(
    data: Union[h5py.Group, h5py.Dataset], froot: h5py.File, depth: int = 0
):
    """
    Recursively interpret a MATLAB variable from an HDF5 file.

    Parameters
    ----------
    data : h5py.Group or h5py.Dataset
        The HDF5 dataset or group representing the MATLAB variable.
    froot : h5py.File
        The root HDF5 file from which the variable is read.

    Returns
    -------
    value : object
        The interpreted MATLAB variable, which can be a numpy array, structured array,
        sparse matrix, or other types depending on the MATLAB class.
    """
    if depth > 99:
        raise RecursionError(
            "Maximum recursion depth exceeded while reading MATLAB variable."
        )

    mat_class = data.attrs.get("MATLAB_class")
    if mat_class is not None:
        mat_class = mat_class.decode()
    else:
        mat_class = "cell"

    if "MATLAB_sparse" in data.attrs:
        data = data["data"]
        row = data["ir"]
        col = data["jc"]
        nrows = data.attrs["MATLAB_sparse"]
        ncols = len(col) - 1
        value = csc_matrix((data, row, col), shape=(nrows, ncols))

    elif mat_class == "struct":
        fields = list(data.keys())
        dtype = [(field, "O") for field in fields]
        values = [get_variable(data[field], froot, depth + 1) for field in fields]
        try:
            # if there is the same number of elements in each field, pack into a structured array
            # to mimic MATLAB struct arrays. structured arrays assume data is packed as tuples
            value = list(map(tuple, np.stack(values).T))
            value = np.array(value, dtype=dtype)
        except Exception as e:
            # if the fields have different number of elements, pack into a structured void to mimic MATLAB struct
            if "all input arrays must have the same shape" in e.args[0]:
                value = np.void(tuple(values), dtype=dtype)
            else:
                # if the error is not related to shape, re-raise it
                raise e

    elif mat_class == "cell":
        shape = data[:].shape
        fr = data[:].ravel()
        value = np.empty(fr.shape, dtype=object)
        for idx, ref in enumerate(fr):
            value[idx] = get_variable(froot[ref], froot, depth + 1)
        value = np.squeeze(value.reshape(shape).T)

    elif mat_class in (
        "logical",
        "double",
        "single",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    ):
        value = np.squeeze(data[:].T)
        if mat_class == "logical":
            value = value.astype(bool)
        if len(value.shape) == 0:
            value = value.item()

    elif mat_class == "char":
        chars = np.vectorize(chr)(data[:]).T
        value = np.squeeze(np.array(["".join(c) for c in chars]))
        if len(value.shape) == 0:
            value = value.item()

    else:
        print(data, mat_class)
        raise ValueError(f"Unsupported MATLAB class: {mat_class}")

    return value


def loadmat(fpath: str) -> np.void:
    """
    Load a MATLAB .mat file from an HDF5 format.

    Parameters
    ----------
    fpath : str
        Path to the .mat file.

    Returns
    -------
    np.void
        A structured void containing the MATLAB variables for use by the MatStruct class.

    """
    with h5py.File(fpath, "r") as f:

        keys = list(f.keys())
        if "#refs#" in keys:
            keys.remove("#refs#")

        dtype = [(field, "O") for field in keys]
        values = [get_variable(f[field], f) for field in keys]

    # file level struct is a structured void
    return np.void(tuple(values), dtype=dtype)
