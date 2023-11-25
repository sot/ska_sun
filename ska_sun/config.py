# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Configuration for ska_sun.

See https://docs.astropy.org/en/stable/config/index.html#customizing-config-location-in-affiliated-packages
and https://github.com/astropy/astropy/issues/12960.
"""  # noqa

from astropy import config
from astropy.config import ConfigNamespace


class ConfigItem(config.ConfigItem):
    rootname = "ska_sun"


class Conf(ConfigNamespace):
    """
    Configuration parameters for ska_sun.
    """

    sun_position_method_default = ConfigItem(
        ["fast_and_accurate", "fast", "accurate"],
        "Default value of `method` parameter in ska_sun.position()"
        ' (default="fast_and_accurate").',
    )

    fast_and_accurate_pitch_limit = ConfigItem(
        165.0,
        "Pitch value above which the accurate method is used for "
        "ska_sun.pitch() and ska_sun.off_nom_roll() when method='fast_and_accurate'.",
    )

    from_chandra_default = ConfigItem(
        False,
        "Default value of `from_chandra` parameter in ska_sun.position_accurate() "
        "(default=False).",
    )


# Create a configuration instance for the user
conf = Conf()
