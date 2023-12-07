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
        ["accurate", "fast"],
        "Default value of `method` parameter in ska_sun.position()"
        ' (default="accurate").',
    )


# Create a configuration instance for the user
conf = Conf()
