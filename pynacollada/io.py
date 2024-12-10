import os, shutil, glob
import pooch
from pooch import Pooch
import dandi
import fsspec
import h5py
from dandi.dandiapi import DandiAPIClient
from fsspec.implementations.cached import CachingFileSystem
from pynwb import NWBHDF5IO
from pynacollada.settings import config
from pathlib import Path
from .loaders import dandi_downloader, load_nwb

DATA_REGISTRY = {
    "mesoscale_activity": {
        "sub-480928_ses-20210129T132207_behavior+ecephys+ogen.nwb": "3526a812f126fe09205c4ef5592974cce78bab5625f3eacf680165e35d56b443",
        # "sub-484672_ses-20210321T131204_behavior+ecephys+ogen.nwb": "e5017930d129797d51fc1cfa3f434d51f54988f1f726f8ed121138b56f2291b3", # smaller file for testing
    },
    "perceptual_straightening": {
        "ay5_u002_image_sequences.mat": "c1b8a03e624a1e79b6c8c77fb3f9d83cd6fc9ee364f5ed334883bbc81c38ca0f",
        "stim_info.mat": "a7880cd0a0321d72c82f0639078aa017b9249a2bd90320c19182cd0ee34de890",
        "stim_matrix.mat": "910f4ac5a5a8b2ffd6ed165a9cd50260663500cd17ed69a547bca1f1ae3290fb",
    },
}

"https://api.dandiarchive.org/api/assets/d524f0d4-6f5c-4d74-8f99-094e360579c5/download/"

DATA_URLS = {
    "mesoscale_activity": {
        "sub-480928_ses-20210129T132207_behavior+ecephys+ogen.nwb": "https://api.dandiarchive.org/api/assets/3d142f75-f3c0-4106-9533-710d26f12b02/download/",
        # "sub-484672_ses-20210321T131204_behavior+ecephys+ogen.nwb": "https://api.dandiarchive.org/api/assets/ad207ee4-8f59-47f3-9201-005d933b7ac1/download/"
    },
    "perceptual_straightening": {
        "ay5_u002_image_sequences.mat": "https://osf.io/9kbnw/download",
        "stim_info.mat": "https://osf.io/gwtcs/download",
        "stim_matrix.mat": "https://osf.io/bh6mu/download",
    },
}


DATA_DOWNLOADER = {
    "mesoscale_activity": dandi_downloader,
    "perceptual_straightening": None,
}

# DATA_LOADER = {
#     "mesoscale_activity": load_nwb,
#     "perceptual_straightening": load_mat,
# }

# dataset = "mesoscale_activity"


def fetch_data(dataset, stream_data=False):
    """
    Fetches tutorial data from disk, or downloads it if it doesn't exist.

    Parameters
    ----------
    dataset : str
        Name of the tutorial whose data should be fetched.
    stream_data : bool
        Whether to stream the data instead of downloading directly. Only works for DANDI datasets. Defaults to False.

    Returns
    -------


    """

    manager = pooch.create(
        path=config["data_dir"] / dataset,
        base_url="",
        urls=DATA_URLS[dataset],
        registry=DATA_REGISTRY[dataset],
        retry_if_failed=2,
        allow_updates="POOCH_ALLOW_UPDATES",
    )

    files = []
    for key in DATA_URLS[dataset]:
        files.append(
            manager.fetch(
                key,
                progressbar=True,
                downloader=DATA_DOWNLOADER[dataset],
            )
        )

    return files


# def load_data(dataset, stream_data=False):
#     """
#     Load tutorial data from disk, or download it if it doesn't exist.

#     Parameters
#     ----------
#     dataset : str
#         Name of the tutorial whose data should be fetched.
#     stream_data : bool
#         Whether to stream the data instead of downloading directly. Only works for DANDI datasets. Defaults to False.

#     Returns
#     -------


#     """
#     files = fetch_data(dataset, stream_data)
#     return DATA_LOADER[dataset](files)
