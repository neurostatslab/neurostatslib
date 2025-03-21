import argparse
from .settings import config
from .registry import DATA_REGISTRY
from .io import fetch_data, download_notebook


def parse_neurostatslib_args(args=None):
    """
    Command line arguments for the neurostatslib package.
    """
    parser = argparse.ArgumentParser(
        description="Download or manage neuroscience data analysis tutorials provided by the neurostatslib package. "
        "On the first run, a local configuration file 'neurstatslib_conf.json' is created in the current working "
        "directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        "--data-dir",
        type=str,
        default=config["data_dir"],
        help="Path to the directory where data is stored. "
        "When set, the setting is saved to the local configuration file.",
        dest="data_dir",
    )
    parser.add_argument(
        "--notebook_dir",
        "--notebook-dir",
        type=str,
        default=config["notebook_dir"],
        help="Path to the directory where notebooks are stored. "
        "When set, the setting is saved to the local configuration file.",
        dest="notebook_dir",
    )
    parser.add_argument(
        "--notebook_source",
        "--notebook-source",
        type=str,
        default=config["notebook_source"],
        help="GitHub branch to pull the notebooks. When set, the setting is saved the a local configuration file.",
        dest="notebook_source",
    )
    parser.add_argument(
        "--download",
        "-d",
        choices=list(DATA_REGISTRY.keys()),
        help="Download a specific dataset and tutorial notebook to the data and notebook directories.",
        dest="download",
    )
    parser.add_argument(
        "--config",
        "-c",
        action="store_true",
        help="Print the current configuration settings.",
        dest="config",
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
    config.save()

    if args.config:
        print("Current configuration settings for neurostatslib:")
        for key, value in config.items():
            print(f"  --{key}: {value}")

    if args.download:
        fetch_data(args.download)
        download_notebook(args.download)
