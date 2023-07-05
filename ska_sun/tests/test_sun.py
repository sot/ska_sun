# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest
from Quaternion import Quat

from ska_sun.sun import (
    allowed_rolldev,
    apply_sun_pitch_yaw,
    get_sun_pitch_yaw,
    nominal_roll,
    off_nominal_roll,
)
from ska_sun.sun import pitch as sun_pitch
from ska_sun.sun import position

# Expected pitch, rolldev pairs
exp_pitch_rolldev = np.array(
    [
        [0, -1.0],
        [40, -1.0],
        [56.0, -1.0],
        [85.49229, 4.95597],
        [85.52, 4.9560848],
        [124.99, 12.23652],
        [125, 12.2380],
        [135, 13.979],
        [138, 14.516],
        [177.9, 18.749],
        [177.997, 18.749],
        [178.0, -1.0],
        [180, -1.0],
        [181, -1.0],
    ]
)


@pytest.mark.parametrize("pitch, rolldev", exp_pitch_rolldev)
def test_allowed_rolldev(pitch, rolldev, monkeypatch):
    monkeypatch.setenv("CHANDRA_MODELS_DEFAULT_VERSION", "3.49")
    # Test array of pitchs and allowed roll dev
    assert np.isclose(allowed_rolldev(pitch), rolldev)


def test_allowed_rolldev_vector(monkeypatch):
    monkeypatch.setenv("CHANDRA_MODELS_DEFAULT_VERSION", "3.49")
    assert np.allclose(
        allowed_rolldev(exp_pitch_rolldev[:, 0]), exp_pitch_rolldev[:, 1]
    )


def test_position():
    ra, dec = position("2008:002:00:01:02")
    assert np.allclose((ra, dec), (281.903448, -22.989273))


def test_nominal_roll():
    roll = nominal_roll(205.3105, -14.6925, time="2011:019:20:51:13")
    assert np.allclose(roll, 68.83020)  # vs. 68.80 for obsid 12393 in JAN1711A


def test_off_nominal_roll_and_pitch():
    att = (198.392135, 36.594359, 33.983322)  # RA, Dec, Roll of obsid 16354
    oroll = off_nominal_roll(att, "2015:335:00:00:00")  # NOT the 16354 time
    assert np.isclose(oroll, -12.224010)

    date = "2015:077:01:07:04"
    pitch = sun_pitch(att[0], att[1], time=date)
    assert np.isclose(pitch, 139.5651)  # vs. 139.59 in SOT MP page

    pitch, _ = get_sun_pitch_yaw(att[0], att[1], time=date)
    assert np.isclose(pitch, 139.5651)  # vs. 139.59 in SOT MP page


def test_apply_get_sun_pitch_yaw():
    """Test apply and get sun_pitch_yaw with multiple components"""
    att = apply_sun_pitch_yaw(
        [0, 45, 0], pitch=[0, 10, 20], yaw=[0, 5, 10], sun_ra=0, sun_dec=90
    )
    pitch, yaw = get_sun_pitch_yaw(att.ra, att.dec, sun_ra=0, sun_dec=90)
    assert np.allclose(pitch, 45 + np.array([0, 10, 20]))
    assert np.allclose(yaw, 180 + np.array([0, 5, 10]))


def test_apply_sun_pitch_yaw():
    """Basic test of apply_sun_pitch_yaw"""
    att = Quat(equatorial=[0, 45, 0])
    att2 = apply_sun_pitch_yaw(att, pitch=10, yaw=0, sun_ra=0, sun_dec=0)
    assert np.allclose((att2.ra, att2.dec), (0, 55))

    att2 = apply_sun_pitch_yaw(att, pitch=0, yaw=10, sun_ra=0, sun_dec=90)
    assert np.allclose((att2.ra, att2.dec), (10, 45))


def test_apply_sun_pitch_yaw_with_grid():
    """Use np.ogrid to make a grid of RA/Dec values (via dpitches and dyaws)"""
    dpitches, dyaws = np.ogrid[0:-3:2j, -5:5:3j]
    atts = apply_sun_pitch_yaw(
        att=[1, 45, 10], pitch=dpitches, yaw=dyaws, sun_ra=0, sun_dec=90
    )
    assert atts.shape == (2, 3)
    exp = np.array(
        [
            [[356.0, 45.0, 10.0], [1.0, 45.0, 10.0], [6.0, 45.0, 10.0]],
            [[356.0, 48.0, 10.0], [1.0, 48.0, 10.0], [6.0, 48.0, 10.0]],
        ]
    )
    assert np.allclose(atts.equatorial, exp)


def test_get_sun_pitch_yaw():
    """Test that values approximately match those from ORviewer.

    See slack discussion "ORviewer sun / anti-sun plots azimuthal Sun yaw angle"
    """
    pitch, yaw = get_sun_pitch_yaw(109, 55.3, time="2021:242")
    assert np.allclose((pitch, yaw), (60.453385, 29.880125))
    pitch, yaw = get_sun_pitch_yaw(238.2, -58.9, time="2021:242")
    assert np.allclose((pitch, yaw), (92.405603, 210.56582))
    pitch, yaw = get_sun_pitch_yaw(338, -9.1, time="2021:242")
    assert np.allclose((pitch, yaw), (179.417797, 259.703451))


def test_roll_table_meta():
    from ska_sun.sun import ROLL_TABLE

    # A sampling of args from the roll table meta
    exp = {
        "file_path": "chandra_models/pitch_roll/pitch_roll_constraint.csv",
        "version": None,
        "repo_path": "None",
        "require_latest_version": False,
        "timeout": 5,
    }
    for key, val in exp.items():
        assert ROLL_TABLE.val.meta["call_args"][key] == val
