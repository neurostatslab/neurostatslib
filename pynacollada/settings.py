import os
import json
import pooch
from pathlib import Path
from os import PathLike

LOCAL_CONFIG = "nsl_tutorials_conf.json"

# try subdirectory data, otherwise default to working directory
DEFAULT_DIR = pooch.os_cache("pynacollada")
defaults = dict(
    {
        "data_dir": DEFAULT_DIR,
    }
)


class Config:
    """
    Configuration settings for neurotutorials package. Can be updated and saved to a local configuration file, 'nsl_tutorials_conf.json', in the current working directory.

    If a local configuration file is found in the current working directory, it will be loaded automatically when the package is imported.

    Global configuration files are not supported. Use the 'load' method to load a configuration file from a different location or filename.

    Attributes
    ----------
    data_dir : str
        Path to data directory. Defaults to subdirectory 'data' if it exists in the current working directory, otherwise defaults to the current working directory.

    Examples
    --------
    View current configuration settings:
    >>> import neurotutorials as nt
    >>> nt.config
    {'data_dir': '/path/to/data'}

    Update configuration settings:
    1. Using `update` method
    >>> nt.config.update({'data_dir': '/path/to/new/data'})
    >>> nt.config
    {'data_dir': '/path/to/new/data'}

    2. Using dictionary-like syntax
    >>> nt.config['data_dir'] = '/path/to/newer/data'
    >>> nt.config
    {'data_dir': '/path/to/newer/data'}

    3. Update as an attribute
    >>> nt.config.data_dir = '/path/to/newest/data'
    >>> nt.config
    {'data_dir': '/path/to/newest/data'}
    """

    _instance = None

    # override __new__ to enforce a single instance of Config
    def __new__(cls, conf_file=LOCAL_CONFIG, defaults=defaults):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            if os.path.exists(conf_file):
                import json

                with open(conf_file, "r") as f:
                    conf = json.load(f)
            else:
                conf = defaults

            cls._instance.update(conf)

        return cls._instance

    def __setitem__(self, key, value):
        self.update({key: value})

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return self.__dict__.__repr__()

    @classmethod
    def _validate_conf(cls, conf):
        if not isinstance(conf, dict):
            raise ValueError("Configuration must be a dictionary")
        if "data_dir" in conf:
            if isinstance(conf["data_dir"], str):
                # convert string to path
                conf["data_dir"] = Path(conf["data_dir"])
            if not isinstance(conf["data_dir"], PathLike):
                raise ValueError("data_dir must be a string or PathLike object")

    def update(self, conf):
        """
        Update configuration settings.

        Parameters
        ----------
        conf : dict
            Dictionary of configuration settings.
        """
        Config._validate_conf(conf)
        self.__dict__.update(conf)

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


config = Config()
