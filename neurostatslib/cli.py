import argparse
from .settings import config
from .registry import DATA_REGISTRY
from .io import fetch_data, download_notebook


def parse_neurostatslib_args(args=None):
    """
    Command line arguments for the neurostatslib package.
    """
    parser = argparse.ArgumentParser(
        description="Neuroscience data analysis tutorials by NeuroStatsLab using publicly available, curated data sets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        type=str,
        default=config["data_dir"],
        help="Path to the directory where data is stored.",
        dest="data_dir",
    )
    parser.add_argument(
        "--notebook-dir",
        "--notebook_dir",
        type=str,
        default=config["notebook_dir"],
        help="Path to the directory where notebooks are stored.",
        dest="notebook_dir",
    )
    parser.add_argument(
        "--notebook-source",
        "--notebook_source",
        type=str,
        default=config["notebook_source"],
        help="GitHub branch to pull the notebooks.",
        dest="notebook_source",
    )
    parser.add_argument(
        "--save",
        "-s",
        action="store_true",
        help="Save the configuration settings to a local file.",
    )
    parser.add_argument(
        "--download-data",
        "--download_data",
        choices=["all"] + list(DATA_REGISTRY.keys()),
        help="Download a specific dataset or all datasets.",
    )
    parser.add_argument(
        "--download-notebook",
        "--download_notebook",
        choices=["all"] + list(DATA_REGISTRY.keys()),
        help="Download a specific notebook or all notebooks.",
    )
    return parser.parse_args(args)


def neurostatslib(args=None):
    """
    Entry point for the neurostatslib script.
    """
    args = parse_neurostatslib_args(args)
    config["data_dir"] = args.data_dir
    config["notebook_dir"] = args.notebook_dir
    config["notebook_source"] = args.notebook_source

    if args.save:
        config.save()

    if args.download_data:
        if args.download_data == "all":
            for dataset in DATA_REGISTRY.keys():
                fetch_data(dataset)
        else:
            fetch_data(args.download_data)

    if args.download_notebook:
        if args.download_notebook == "all":
            for dataset in DATA_REGISTRY.keys():
                download_notebook(dataset)
        else:
            download_notebook(args.download_notebook)

    return config
