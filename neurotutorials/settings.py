import os
import json

LOCALCONFIG = "nsl_tutorials_conf.json"

# try subdirectory data, otherwise default to working directory
data_path = os.path.join(os.getcwd(), "data")
defaults = dict(
    {
        "data_path": data_path if os.path.exists(data_path) else os.getcwd(),
    }
)


class Config:
    """
    Configuration settings for neurotutorials package. Can be updated and saved to a local configuration file in the current working directory.

    If a local configuration file is found in the current working directory, it will be loaded automatically when the package is imported. Global configuration files are not supported.

    Attributes
    ----------
    data_path : str
        Path to data directory.

    """

    _instance = None

    # override __new__ to enforce a single instance of Config
    def __new__(cls, conf=LOCALCONFIG, defaults=defaults):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            if os.path.exists(conf):
                import json

                with open(conf, "r") as f:
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
        if not os.path.exists(conf["data_path"]):
            raise ValueError(f"Data path {conf['data_path']} does not exist")

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

    def save(self, conf=LOCALCONFIG):
        """
        Save a local configuration file.
        """
        with open(conf, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def load(self, conf=LOCALCONFIG):
        with open(conf, "r") as f:
            self.update(json.load(f))


config = Config()
