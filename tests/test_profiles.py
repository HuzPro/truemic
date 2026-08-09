import pytest

from truemic import config


@pytest.fixture(autouse=True)
def restore_config():
    saved = {name: getattr(config, name) for name in config.PROFILES["low-power"]}
    yield
    for name, value in saved.items():
        setattr(config, name, value)


def test_low_power_profile_slows_the_decide_cadence():
    config.apply_profile("low-power")
    assert config.DECIDE_EVERY_SECONDS > 0.5
    assert config.WM_CHUNK_SECONDS >= 1.0


def test_low_power_profile_slows_the_detect_hop():
    config.apply_profile("low-power")
    assert config.DETECT_HOP_SECONDS >= 1.0


def test_unknown_profile_is_rejected():
    with pytest.raises(KeyError):
        config.apply_profile("turbo")


def test_default_profile_leaves_settings_alone():
    before = config.DECIDE_EVERY_SECONDS
    config.apply_profile("default")
    assert config.DECIDE_EVERY_SECONDS == before
