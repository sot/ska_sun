# Licensed under a 3-clause BSD style license - see LICENSE.rst

import ska_helpers

from .sun import *  # noqa

__version__ = ska_helpers.get_version('ska_sun')


def test(*args, **kwargs):
    '''
    Run py.test unit tests.
    '''
    import testr

    return testr.test(*args, **kwargs)
