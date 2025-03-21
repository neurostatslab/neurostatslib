from .registry import (
    DATA_REGISTRY,
    DATA_URLS,
    DATA_DOWNLOADER,
    DATA_LOADER,
    NOTEBOOK_REGISTRY,
)
import jupytext
from .settings import config
import pooch
from .utils import flatten
import os


def data_manager():
    """
    Pooch manager for downloading and managing tutorial data.
    """
    return pooch.create(
        path=config["data_dir"],
        base_url="",
        registry=dict(flatten(DATA_REGISTRY)),
        urls=dict(flatten(DATA_URLS)),
        retry_if_failed=2,
        allow_updates="POOCH_ALLOW_UPDATES",
    )


def notebook_manager():
    """
    Pooch manager for downloading and managing tutorial notebook mardown files from GitHub.
    """
    return pooch.create(
        path=config["data_dir"] + "/notebooks",
        base_url=f"https://raw.github.com/neurostatslab/neurostatslib/{config['notebook_source']}/docs/data_sets/",
        registry=NOTEBOOK_REGISTRY,
        retry_if_failed=2,
        allow_updates="POOCH_ALLOW_UPDATES",
    )


def fetch_data(dataset, stream_data=False):
    """
    Fetches tutorial data from disk, or downloads it if it doesn't exist.
    Data are downloaded to the data directory specified in the configuration settings.

    Parameters
    ----------
    dataset : str
        Name of the tutorial whose data should be fetched.
    stream_data : bool
        Whether to stream the data instead of downloading directly. Only works for DANDI datasets. Defaults to False.

    Returns
    -------
    files : list
        List of paths to downloaded files

    """
    manager = data_manager()
    project = {dataset + "/" + k: v for k, v in DATA_URLS[dataset].items()}
    files = []
    for key in project.keys():
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
    Data are downloaded to the data directory specified in the configuration settings.

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


def fetch_notebook(dataset):
    """
    Fetches tutorial notebook markdown file from disk, or downloads it if it doesn't exist.

    Parameters
    ----------
    dataset : str
        Name of the tutorial whose notebook should be fetched.

    Returns
    -------
    fpath : str
        Path to the downloaded notebook

    """

    manager = notebook_manager()
    fpath = manager.fetch(
        dataset + ".md",
        progressbar=True,
    )

    return fpath


def download_notebook(dataset, overwrite=False):
    """
    Downloads a tutorial notebook from GitHub to the notebook directory.

    Under the hood, this function fetches the notebook markdown file from GitHub and converts it to a Jupyter notebook,
    where the jupyter notebook is saved to the notebook directory.

    Parameters
    ----------
    dataset : str
        Name of the tutorial whose notebook should be downloaded.
    overwrite : bool
        Whether to overwrite the notebook if it already exists. Defaults to False.
        Overwriting mean reconverting the Jupyter notebook from the markdown file.
    """

    if os.path.exists(config["notebook_dir"] + "/" + dataset + ".ipynb"):
        if overwrite is False:
            print(
                f"Notebook already exists at {config['notebook_dir']}/{dataset}.ipynb\n"
                "Set overwrite=True to download again."
            )
            return
        else:
            print(f"Overwriting existing notebook {dataset}.ipynb")
            os.remove(config["notebook_dir"] + "/" + dataset + ".ipynb")

    if os.path.exists(config["notebook_dir"]) is False:
        os.mkdir(config["notebook_dir"])

    fpath = fetch_notebook(dataset)
    notebook = jupytext.read(fpath)
    jupytext.write(notebook, config["notebook_dir"] + "/" + dataset + ".ipynb")
    print(f"Downloaded notebook to {config['notebook_dir']}/{dataset}.ipynb")
