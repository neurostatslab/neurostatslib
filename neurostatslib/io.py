from .registry import DATA_REGISTRY, DATA_URLS, DATA_DOWNLOADER, DATA_LOADER
from .settings import config
import pooch
from .utils import flatten


MANAGER = pooch.create(
    path=config["data_dir"],
    base_url="",
    registry=dict(flatten(DATA_REGISTRY)),
    urls=dict(flatten(DATA_URLS)),
    retry_if_failed=2,
    allow_updates="POOCH_ALLOW_UPDATES",
)


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

    project = {dataset + "/" + k: v for k, v in DATA_URLS[dataset].items()}
    files = []
    for key in project.keys():
        files.append(
            MANAGER.fetch(
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
