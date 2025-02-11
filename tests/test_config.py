import pynacollada as nac
from pathlib import Path
import copy
import pytest
from contextlib import nullcontext as does_not_raise


def test_config_defaults():
    defaults = nac.settings.defaults

    for key, value in defaults.items():
        assert nac.config[key] == value


def test_config_reset():
    nac.config["data_dir"] = "/path/to/data"
    nac.config["unique_data_dir"]["perceptual_straightening"] = "/other/path/to/data"
    nac.config["user"] = "test_user"

    for key in ["data_dir", "unique_data_dir", "user"]:
        assert key in nac.config.keys()

    nac.config.reset()

    for key, value in nac.settings.defaults.items():
        assert nac.config[key] == value

    assert "user" not in nac.config.keys()


@pytest.fixture
def clear_config():
    nac.config.reset()


@pytest.mark.usefixtures("clear_config")
class Test_Config:
    @pytest.mark.parametrize(
        "key, value, expected",
        [
            ("data_dir", "/path/to/data", does_not_raise()),
            ("data_dir", Path("test"), does_not_raise()),
            ("data_dir", {}, pytest.raises(TypeError)),
            ("data_dir", 123, pytest.raises(TypeError)),
            (
                "unique_data_dir",
                {"perceptual_straightening": "/path/to/data"},
                does_not_raise(),
            ),
            (
                "unique_data_dir",
                {"perceptual_straightening": Path("test")},
                does_not_raise(),
            ),
            (
                "unique_data_dir",
                {"perceptual_straightening": {}},
                pytest.raises(TypeError),
            ),
            (
                "unique_data_dir",
                {"perceptual_straightening": 123},
                pytest.raises(TypeError),
            ),
            (
                "unique_data_dir",
                "test",
                pytest.raises(TypeError, match="must be a dictionary"),
            ),
            (
                "unique_data_dir",
                {"test": "test"},
                pytest.raises(ValueError, match="Invalid dataset name"),
            ),
            (1, "test", pytest.raises(TypeError, match="Key must be a string")),
        ],
    )
    def test_config_update(self, key, value, expected):
        # test update with setitem
        with expected as e:
            nac.config[key] = value

        if not e:
            if isinstance(value, dict):
                for k, v in value.items():
                    assert nac.config[key][k] == str(v)
            else:
                assert nac.config[key] == str(value)

        nac.config.reset()

        # test update with update method
        with expected as e:
            nac.config.update({key: value})

        if not e:
            if isinstance(value, dict):
                for k, v in value.items():
                    assert nac.config[key][k] == str(v)
            else:
                assert nac.config[key] == str(value)

    @pytest.mark.parametrize(
        "conf, expected",
        [
            (
                {
                    "data_dir": "/path/to/data",
                    "unique_data_dir": {"perceptual_straightening": "/path/to/data"},
                },
                does_not_raise(),
            ),
            (
                {
                    "data_dir": Path("test"),
                    "unique_data_dir": {"perceptual_straightening": Path("test")},
                },
                does_not_raise(),
            ),
            (
                {
                    "data_dir": {},
                    "unique_data_dir": {"perceptual_straightening": "test"},
                },
                pytest.raises(TypeError),
            ),
            (
                {
                    "data_dir": "/path/to/data",
                    "unique_data_dir": "",
                },
                pytest.raises(TypeError),
            ),
            (
                {
                    "data_dir": "/path/to/data",
                    "unique_data_dir": {"test": "test"},
                },
                pytest.raises(ValueError, match="Invalid dataset name"),
            ),
        ],
    )
    def test_config_update_multiple(self, conf, expected):
        with expected as e:
            nac.config.update(conf)
        if not e:
            for key, value in conf.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        assert nac.config[key][k] == str(v)
                else:
                    assert nac.config[key] == str(value)

    def test_config_update_kwargs(self):
        nac.config.update(data_dir="/path/to/newest/data")
        assert nac.config["data_dir"] == "/path/to/newest/data"

        with pytest.raises(TypeError):
            nac.config.update(data_dir=123)

        nac.config.update(unique_data_dir={"perceptual_straightening": "/path/to/data"})
        assert (
            nac.config["unique_data_dir"]["perceptual_straightening"] == "/path/to/data"
        )

        with pytest.raises(TypeError):
            nac.config.update(unique_data_dir={"perceptual_straightening": 123})

        with pytest.raises(ValueError):
            nac.config.update(unique_data_dir={"test": "test"})

        nac.config.update(data_dir="/other/path/to/data", unique_data_dir={})
        assert nac.config["data_dir"] == "/other/path/to/data"
        assert nac.config["unique_data_dir"] == {}

    # @pytest.mark.parameterize("path", [nac.settings.LOCAL_CONFIG, "test_conf.json"])
    def test_config_save_and_load(self):
        # store defaults
        config_old = copy.deepcopy(dict(nac.config))

        # update config
        nac.config.update(
            data_dir="/other/path/to/data",
            unique_data_dir={"perceptual_straightening": "/another/path/to/data"},
            user="test_user",
        )

        # store new config
        config_new = copy.deepcopy(dict(nac.config))

        # save, reset, and reload config
        nac.config.save()
        assert Path(nac.settings.LOCAL_CONFIG).exists()
        nac.config.reset()
        nac.config.load()

        # assert that the loaded config is the same as the saved config
        assert nac.config["data_dir"] == config_new["data_dir"]
        assert (
            nac.config["unique_data_dir"]["perceptual_straightening"]
            == config_new["unique_data_dir"]["perceptual_straightening"]
        )
        assert nac.config["user"] == config_new["user"]

        # assert that the loaded config is different from the original
        assert nac.config["data_dir"] != config_old["data_dir"]
        assert (
            nac.config["unique_data_dir"]["perceptual_straightening"]
            != config_old["unique_data_dir"]["perceptual_straightening"]
        )
        assert "user" not in config_old.keys()

        # save a second new config at a custom path
        path = "test_conf.json"
        nac.config.update(
            data_dir="/another/path/to/data",
            unique_data_dir={"perceptual_straightening": "/other/path/to/data"},
            user="other_test_user",
        )
        config_new_2 = copy.deepcopy(dict(nac.config))

        # save and load previous config at default location
        nac.config.save(path)
        assert Path(path).exists()
        nac.config.load()

        # make sure we loaded in the first new config
        assert nac.config["data_dir"] == config_new["data_dir"]
        assert (
            nac.config["unique_data_dir"]["perceptual_straightening"]
            == config_new["unique_data_dir"]["perceptual_straightening"]
        )
        assert nac.config["user"] == config_new["user"]

        # load in the second new config
        nac.config.load(path)

        # assert that the loaded config is the same as the second saved config
        assert nac.config["data_dir"] == config_new_2["data_dir"]
        assert (
            nac.config["unique_data_dir"]["perceptual_straightening"]
            == config_new_2["unique_data_dir"]["perceptual_straightening"]
        )
        assert nac.config["user"] == config_new_2["user"]

        # assert that the loaded config is different from the first saved config
        assert nac.config["data_dir"] != config_new["data_dir"]
        assert (
            nac.config["unique_data_dir"]["perceptual_straightening"]
            != config_new["unique_data_dir"]["perceptual_straightening"]
        )
        assert nac.config["user"] != config_new["user"]

        # remove the config files
        Path(nac.settings.LOCAL_CONFIG).unlink()
        assert not Path(nac.settings.LOCAL_CONFIG).exists()

        Path(path).unlink()
        assert not Path(path).exists()
