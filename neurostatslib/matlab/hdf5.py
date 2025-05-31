import h5py
from scipy.sparse import csc_matrix
import numpy as np

# class MatFileReader:


def get_variable(f, ffull):

    mat_class = f.attrs.get("MATLAB_class")
    if mat_class is not None:
        mat_class = mat_class.decode()
    else:
        mat_class = "cell"

    if "MATLAB_sparse" in f.attrs:
        data = f["data"]
        row = f["ir"]
        col = f["jc"]
        nrows = f.attrs["MATLAB_sparse"]
        ncols = len(col) - 1
        value = csc_matrix((data, row, col), shape=(nrows, ncols))

    elif mat_class == "struct":
        fields = list(f.keys())
        dtype = []
        values = []
        for field in fields:
            dtype.append((field, "O"))
            values.append(get_variable(f[field], ffull))
        try:
            value = np.array(values, dtype=dtype)
        except Exception as e:
            if ("Cannot cast array data" in e.args[0]) or (
                "setting an array element with a sequence" in e.args[0]
            ):
                value = np.void(tuple(values), dtype=dtype)
            else:
                raise e

    elif mat_class == "cell":
        shape = f[:].shape
        fr = f[:].ravel()
        value = np.empty(fr.shape, dtype=object)
        for idx, ref in enumerate(fr):
            value[idx] = get_variable(ffull[ref], ffull)
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
        value = np.squeeze(f[:].T)
        if mat_class == "logical":
            value = value.astype(bool)
        if len(value.shape) == 0:
            value = value.item()

    elif mat_class == "char":
        chars = np.vectorize(chr)(f[:]).T
        value = np.array(["".join(c) for c in chars])

    else:
        print(f, mat_class)

    return value


def loadmat(fpath):
    f = h5py.File(fpath, "r")

    # data = {}
    keys = list(f.keys())
    if "#refs#" in keys:
        # data['refs'] = f['#refs#']
        keys.remove("#refs#")

    dtype = []
    values = []
    for key in keys:
        dtype.append((key, "O"))
        values.append(get_variable(f[key], f))

    return np.void(tuple(values), dtype=dtype)
