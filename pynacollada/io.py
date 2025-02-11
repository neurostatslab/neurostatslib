from pathlib import Path
import pooch
from .registry import DATA_REGISTRY, DATA_URLS, DATA_DOWNLOADER, DATA_LOADER

from .settings import config


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

    if dataset in config["unique_data_dir"].keys():
        data_dir = config["unique_data_dir"][dataset]
    else:
        data_dir = Path(config["data_dir"]) / dataset

    manager = pooch.create(
        path=data_dir,
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


def load_data(dataset, stream_data=False):
    """
    Load tutorial data from disk, or download it if it doesn't exist.

    Parameters
    ----------
    dataset : str
        Name of the tutorial whose data should be fetched.
    stream_data : bool
        Whether to stream the data instead of downloading directly. Only works for DANDI datasets. Defaults to False.

    Returns
    -------


    """
    files = fetch_data(dataset, stream_data)
    return DATA_LOADER[dataset](files)
