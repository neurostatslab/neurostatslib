from dandi.download import download as dandi_download
import os
import shutil
import glob
from pathlib import Path
from pynwb import NWBHDF5IO
import mat73
import pynapple as nap
from .containers import mat_container
import scipy


# dataset = "mesoscale_activity"
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
    Wrapper for loading a NWB file, using either a string input with the file path, or a length-1 list of a string with
    the file path.
    """

    if isinstance(file_path, list):
        if len(file_path) == 1:
            file_path = file_path[0]
        else:
            raise ValueError("Only one nwb file can be loaded at a time.")
    io = NWBHDF5IO(file_path, "r")
    return io.read()


def nap_load(file_path):
    if isinstance(file_path, list):
        if len(file_path) == 1:
            file_path = file_path[0]
        else:
            raise ValueError("Only one nwb file can be loaded at a time.")
    return nap.load_file(file_path)


def load_mat(file_path, file_name="root"):
    """
    Load in a .mat file or files and returns an extended HDMF container with the data. The HDMF container allows for
    easy visualization of the data and its structure in a Jupyter notebook.
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
