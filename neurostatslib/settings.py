import os
import json
import pooch
from functools import wraps
from os import PathLike
from collections.abc import MutableMapping
from .registry import DATA_REGISTRY, DATA_URLS, NOTEBOOK_REGISTRY
from .utils import flatten

DATA_SETS = DATA_REGISTRY.keys()

LOCAL_CONFIG = "neurostatslib_conf.json"

DEFAULT_DATA_DIR = str(pooch.os_cache("neurostatslib"))
DEFAULT_NOTEBOOK_DIR = os.getcwd() + "/notebooks"
DEFAULTS = dict(
    {
        # where to download the data
        "data_dir": DEFAULT_DATA_DIR,
        # where to download the notebooks
        # this is anticipated for when the package is released on pip and not installed through the repository
        "notebook_dir": DEFAULT_NOTEBOOK_DIR,
        # which branch to download the notebooks from
        "notebook_source": "main",
    }
)


def _validate_conf(func):
    @wraps(func)
    def wrapper(self, key, value):
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
        match key:
            case "data_dir":
                if isinstance(value, PathLike):
                    # convert path to string for serialization
                    value = str(value)
                if not isinstance(value, str):
                    raise TypeError("data_dir must be a string or PathLike object")

            case "notebook_dir":
                if isinstance(value, PathLike):
                    # convert path to string for serialization
                    value = str(value)
                if not isinstance(value, str):
                    raise TypeError("notebook_dir must be a string or PathLike object")

        return func(self, key, value)

    return wrapper


class Config(MutableMapping):
    """
    Configuration settings for neurostatslib package. Can be updated and saved to a local configuration file,
    'neurostatslib_conf.json', in the current working directory.

    If a local configuration file is found in the current working directory, it will be loaded automatically when the
    package is imported.

    Global configuration files are not supported. Use the 'load' method to load a configuration file from a different
    location or filename.

    Attributes
    ----------
    data_dir : str
        Path to data directory. Defaults to subdirectory 'data' if it exists in the current working directory,
        otherwise defaults to the current working directory.

    Examples
    --------
    View current configuration settings:
    >>> import pynacollada as nac
    >>> nac.config
    {'data_dir': '/path/to/data'}

    Update configuration settings:
    1. Using `update` method
    >>> nac.config.update({'data_dir': '/path/to/new/data'})
    >>> nac.config
    {'data_dir': '/path/to/new/data'}
    >>> nac.config.update(data_dir = '/path/to/newer/data')
    >>> nac.config
    {'data_dir': '/path/to/newer/data'}

    2. Using dictionary-like syntax
    >>> nac.config['data_dir'] = '/path/to/newer/data'
    >>> nac.config
    {'data_dir': '/path/to/newer/data'}

    3. Update as an attribute
    >>> nac.config.data_dir = '/path/to/newest/data'
    >>> nac.config
    {'data_dir': '/path/to/newest/data'}
    """

    _instance = None

    # override __new__ to enforce a single instance of Config
    def __new__(cls, conf_file=LOCAL_CONFIG, defaults=DEFAULTS):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.update(defaults)

            if os.path.exists(conf_file):
                import json

                with open(conf_file, "r") as f:
                    conf = json.load(f)

                cls._instance.update(conf)

        return cls._instance

    # def __init__(self, conf_file=LOCAL_CONFIG, defaults=defaults):
    #     super().__init__()
    #     self.update(defaults)

    #     if os.path.exists(conf_file):
    #         import json

    #         with open(conf_file, "r") as f:
    #             conf = json.load(f)

    #         self.update(conf)

    @_validate_conf
    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        return self.__dict__.__repr__()

    def save(self, conf_file=LOCAL_CONFIG):
        """
        Save a local configuration file.

        Parameters
        ----------
        conf_file : str, optional
            Path to save configuration file. Defaults to 'nsl_tutorials_conf.json' in the current working directory.

        Examples
        --------
        Save current configuration settings in the current working directory:
        >>> import neurotutorials as nt
        >>> nt.config["data_dir"] = "/new/path/to/data"
        >>> nt.config.save()

        Save configuration settings to a different location:
        >>> nt.config.save("/path/to/config.json")
        """
        with open(conf_file, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def load(self, conf_file=LOCAL_CONFIG):
        """
        Load settings from a configuration file.

        Parameters
        ----------
        conf_file : str, optional
            Path to configuration file. Defaults to 'nsl_tutorials_conf.json' in the current working directory.

        Examples
        --------
        Load configuration settings from a different location:
        >>> import neurotutorials as nt
        >>> nt.config.load("/path/to/config.json")
        """
        with open(conf_file, "r") as f:
            self.update(json.load(f))

    def reset(self):
        """
        Reset configuration settings to defaults.
        """
        del self.__dict__
        self.update(DEFAULTS)


config = Config()
