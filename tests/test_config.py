import neurostatslib as nsl
from pathlib import Path
import copy
import os
import pytest
from contextlib import nullcontext as does_not_raise


def test_config_defaults():
    defaults = nsl.settings.defaults

    for key, value in defaults.items():
        assert nsl.config[key] == value


def test_config_reset():
    nsl.config["data_dir"] = "/path/to/data"
    nsl.config["user"] = "test_user"

    for key in ["data_dir", "user"]:
        assert key in nsl.config.keys()

    nsl.config.reset()

    for key, value in nsl.settings.defaults.items():
        assert nsl.config[key] == value

    assert "user" not in nsl.config.keys()


def test_config_instance():
    nsl.config.reset()
    nsl.config["data_dir"] = "/path/to/data"
    assert nsl.settings.Config() == nsl.config
    assert id(nsl.settings.Config()) == id(nsl.config)


@pytest.fixture
def clear_config():
    nsl.config.reset()


@pytest.mark.usefixtures("clear_config")
class TestConfig:
    update_args = (
        "key, value, expected",
        [
            ("data_dir", "/path/to/data", does_not_raise()),
            ("data_dir", Path("test"), does_not_raise()),
            ("data_dir", {}, pytest.raises(TypeError)),
            ("data_dir", 123, pytest.raises(TypeError)),
            ("notebook_dir", "/path/to/notebooks", does_not_raise()),
            ("notebook_dir", Path("test"), does_not_raise()),
            ("notebook_dir", {}, pytest.raises(TypeError)),
            ("notebook_dir", 123, pytest.raises(TypeError)),
            ("notebook_source", "test", does_not_raise()),
            ("notebook_source", 123, pytest.raises(TypeError)),
            (1, "test", pytest.raises(TypeError, match="Key must be a string")),
        ],
    )

    @pytest.mark.parametrize(*update_args)
    def test_config_update(self, key, value, expected):
        # test update with update method
        with expected as e:
            nsl.config.update({key: value})

        if not e:
            if isinstance(value, dict):
                for k, v in value.items():
                    assert nsl.config[key][k] == str(v)
            else:
                assert nsl.config[key] == str(value)

    @pytest.mark.parametrize(*update_args)
    def test_config_setitem(self, key, value, expected):
        # test update with setitem
        with expected as e:
            nsl.config[key] = value

        if not e:
            if isinstance(value, dict):
                for k, v in value.items():
                    assert nsl.config[key][k] == str(v)
            else:
                assert nsl.config[key] == str(value)

    def test_config_setattr(self, key, value, expected):
        # test update with setattr
        with expected as e:
            setattr(nsl.config, key, value)

        if not e:
            if isinstance(value, dict):
                for k, v in value.items():
                    assert nsl.config[key][k] == str(v)
            else:
                assert nsl.config[key] == str(value)

    @pytest.mark.parametrize(
        "conf, expected",
        [
            (
                {
                    "data_dir": "/path/to/data",
                    "notebook_dir": "/path/to/notebooks",
                },
                does_not_raise(),
            ),
            (
                {
                    "data_dir": {},
                    "notebook_dir": Path("test"),
                },
                pytest.raises(TypeError),
            ),
            (
                {
                    "data_dir": "/path/to/data",
                    "notebook_dir": {},
                },
                pytest.raises(TypeError),
            ),
        ],
    )
    def test_config_update_multiple(self, conf, expected):
        with expected as e:
            nsl.config.update(conf)
        if not e:
            for key, value in conf.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        assert nsl.config[key][k] == str(v)
                else:
                    assert nsl.config[key] == str(value)

    def test_config_update_kwargs(self):
        nsl.config.update(data_dir="/path/to/newest/data")
        assert nsl.config["data_dir"] == "/path/to/newest/data"

        with pytest.raises(TypeError):
            nsl.config.update(data_dir=123)

    # @pytest.mark.parameterize("path", [nsl.settings.LOCAL_CONFIG, "test_conf.json"])
    def test_config_save_and_load(self):
        # store defaults
        config_old = copy.deepcopy(dict(nsl.config))

        # update config
        nsl.config.update(
            data_dir="/other/path/to/data",
            notebook_dir="other/path/to/notebooks",
            user="test_user",
        )

        # store new config
        config_new = copy.deepcopy(dict(nsl.config))

        # save, reset, and reload config
        nsl.config.save()
        assert Path(nsl.settings.LOCAL_CONFIG).exists()
        nsl.config.reset()
        nsl.config.load()

        # assert that the loaded config is the same as the saved config
        assert nsl.config["data_dir"] == config_new["data_dir"]
        assert nsl.config["notebook_dir"] == config_new["notebook_dir"]
        assert nsl.config["user"] == config_new["user"]

        # assert that the loaded config is different from the original
        assert nsl.config["data_dir"] != config_old["data_dir"]
        assert nsl.config["notebook_dir"] != config_old["notebook_dir"]
        assert "user" not in config_old.keys()

        # save a second new config at a custom path
        path = "test_conf.json"
        nsl.config.update(
            data_dir="/another/path/to/data",
            notebook_dir="/another/path/to/notebooks",
            user="other_test_user",
        )
        config_new_2 = copy.deepcopy(dict(nsl.config))

        # save new config and load previous config at default location
        nsl.config.save(path)
        assert Path(path).exists()
        nsl.config.load()

        # make sure we loaded in the first new config
        assert nsl.config["data_dir"] == config_new["data_dir"]
        assert nsl.config["notebook_dir"] == config_new["notebook_dir"]
        assert nsl.config["user"] == config_new["user"]

        # load in the second new config
        nsl.config.load(path)

        # assert that the loaded config is the same as the second saved config
        assert nsl.config["data_dir"] == config_new_2["data_dir"]
        assert nsl.config["notebook_dir"] == config_new_2["notebook_dir"]
        assert nsl.config["user"] == config_new_2["user"]

        # assert that the loaded config is different from the first saved config
        assert nsl.config["data_dir"] != config_new["data_dir"]
        assert nsl.config["notebook_dir"] != config_new["notebook_dir"]
        assert nsl.config["user"] != config_new["user"]

        # remove the config files
        Path(nsl.settings.LOCAL_CONFIG).unlink()
        assert not Path(nsl.settings.LOCAL_CONFIG).exists()

        Path(path).unlink()
        assert not Path(path).exists()

    def test_parent_config_save_and_load(self):
        # store defaults
        config_old = copy.deepcopy(dict(nsl.config))

        # update config
        nsl.config.update(
            data_dir="/other/path/to/data",
            notebook_dir="other/path/to/notebooks",
            user="test_user",
        )

        # store new config
        config_new = copy.deepcopy(dict(nsl.config))

        # save and reset config
        nsl.config.save()
        assert Path(nsl.settings.LOCAL_CONFIG).exists()
        nsl.config.reset()

        # store a new config with a custom name
        path = "test_conf.json"
        nsl.config.update(
            data_dir="/another/path/to/data",
            notebook_dir="/another/path/to/notebooks",
            user="other_test_user",
        )
        config_new_2 = copy.deepcopy(dict(nsl.config))

        # save and reset
        # save and load previous config at default location
        nsl.config.save(path)
        assert Path(path).exists()
        nsl.config.load()

        # make a child directory and move into it
        os.mkdir("child")
        os.chdir("child")
        assert Path.cwd().stem == "child"

        # load in the first new config
        nsl.config.load()

        # assert that the loaded config is the same as the saved config
        assert nsl.config["data_dir"] == config_new["data_dir"]
        assert nsl.config["notebook_dir"] == config_new["notebook_dir"]
        assert nsl.config["user"] == config_new["user"]

        # assert that the loaded config is different from the original
        assert nsl.config["data_dir"] != config_old["data_dir"]
        assert nsl.config["notebook_dir"] != config_old["notebook_dir"]
        assert "user" not in config_old.keys()

        # load in the second new config
        nsl.config.load(path)

        # assert that the loaded config is the same as the second saved config
        assert nsl.config["data_dir"] == config_new_2["data_dir"]
        assert nsl.config["notebook_dir"] == config_new_2["notebook_dir"]
        assert nsl.config["user"] == config_new_2["user"]

        # assert that the loaded config is different from the first saved config
        assert nsl.config["data_dir"] != config_new["data_dir"]
        assert nsl.config["notebook_dir"] != config_new["notebook_dir"]
        assert nsl.config["user"] != config_new["user"]

        # move back to the parent directory and remove the child directory
        os.chdir("..")
        os.rmdir("child")

        # remove the config files
        Path(nsl.settings.LOCAL_CONFIG).unlink()
        assert not Path(nsl.settings.LOCAL_CONFIG).exists()

        Path(path).unlink()
        assert not Path(path).exists()
