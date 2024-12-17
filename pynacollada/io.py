import os, shutil, glob
from pathlib import Path
from dandi.download import download as dandi_download
from pynwb import NWBHDF5IO
from hdmf.container import Container
import numpy as np
import mat73
import scipy


def dandi_downloader(url, output_file, pooch):
    """
    Custom downloader for dandi files through pooch

    Parameters
    ----------
    url : str
        URL to download the file from
    output_file : str
        Path to save the downloaded file
    pooch : pooch.Pooch
        Pooch object used to download the file. Not used, but needed for compatibility with pooch
    """
    # dandi downloader is faster than pooch for dandi files,
    # so we write a custom downloader that uses dandi
    try:
        dandi_download(url, output_file)
    finally:
        nwbfile = glob.glob(output_file + "/*.nwb*")[0]
        if os.path.isdir(nwbfile):
            # if it's a folder, the download failed or was interrupted
            # echo pooch clean up and remove everything
            # pooch can't do this because it will be a folder
            shutil.rmtree(output_file)
        else:
            # for compatibility with pooch checks, we need to move the dandi downloaded file
            # and rename it to pooch's temp file name, which is currently a folder
            fname = Path(nwbfile).name
            parent = Path(output_file).parent
            shutil.move(nwbfile, parent)  # move out of temp folder
            shutil.rmtree(output_file)  # remove temp folder
            os.rename(parent / fname, output_file)  # rename file to temp name


def load_nwb(file_path):
    """
    Wrapper for loading a NWB file, using either a string input with the file path, or a length-1 list of a string with the file path.
    """

    if isinstance(file_path, list):
        if len(file_path) == 1:
            file_path = file_path[0]
        else:
            raise ValueError("Only one nwb file can be loaded at a time.")
    io = NWBHDF5IO(file_path, "r")
    return io.read()


class MatData(Container):
    """
    HDMF container extended to display properties of array-like data.

    Parameters
    ----------
    name : str
        Name of the data array
    values : np.ndarray
        Values of the data array. Must be a numpy array.

    Attributes
    ----------
    shape : str
        Shape of the data array
    dtype : str
        Data type of the array elements
    values : np.ndarray
        Values of the data array
    """

    __fields__ = (
        "shape",
        "dtype",
        "values",
    )

    def __init__(self, name, values):
        super().__init__(name)
        self.shape = str(np.shape(values))
        # endians don't print for some reason
        self.dtype = str(values.dtype).replace("<", "").replace(">", "")
        self.values = values


class MatField(Container):
    """
    HDMF container extended to display properties of single-valued fields.

    Parameters
    ----------
    name : str
        Name of the field
    value : any
        Value of the field

    Attributes
    ----------
    type : str
        Type of the field
    value : any
        Value of the field
    """

    __fields__ = (
        "type",
        "value",
    )

    def __init__(self, name, value):
        super().__init__(name)
        # endians don't print for some reason
        self.type = type(value).__name__.replace("<", "").replace(">", "")
        self.value = value


def mat_container(struct, name="root"):
    """
    Function to create a container with dynamic fields according to an input dictionary / loaded matlab struct.

    This function is called recursively such that nested structs / dictionaries are represented as nested containers. Arrays are represented as MatData containers, and single values are represented as MatField containers.

    Parameters
    ----------
    struct : dict
        Dictionary or matlab struct loaded with mat73.loadmat or scipy.io.loadmat
    name : str
        Name of the container. On the first call, this will be the name of the outermost container. On nested calls, it will be the dictionary key associated with the conatainer values.
    """

    # recursive function to create nested containers
    def get_container(k, d):
        if isinstance(d, dict):
            return mat_container(d, k)
        elif isinstance(d, (np.ndarray, list)):
            d = np.array(d)
            if len(d.shape):
                return MatData(k, np.array(d))
            elif np.issubdtype(d.dtype, np.number):
                return MatField(k, float(d))
            else:
                return MatField(k, str(d))
        else:
            return MatField(k, d)

    FLDS = struct.keys()

    # HDMF container extended with fields set by keys of the input
    class MatFile(Container):
        __fields__ = tuple(FLDS)

    container = MatFile(name=name)

    # set fields of the container
    for parent_key, parent_data in struct.items():
        setattr(container, parent_key, get_container(parent_key, parent_data))

    return container


def load_mat(file_path, file_name="root"):
    """
    Load in a .mat file or files and returns an extended HDMF container with the data. The HDMF container allows for easy visualization of the data and its structure in a Jupyter notebook.
    """
    if isinstance(file_path, str):
        file_path = [file_path]

    data = {}
    for f in file_path:
        try:
            # mat files saved in >v7.3 format are hdf5 files and handled by mat73
            data[Path(f).stem] = mat73.loadmat(f)
        except TypeError:
            # older mat files are handled by scipy
            data[Path(f).stem] = scipy.io.loadmat(f)

    return mat_container(data, file_name)
