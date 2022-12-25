# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from Quaternion import Quat

from ..sun import (
    allowed_rolldev,
    apply_sun_pitch_yaw,
    get_sun_pitch_yaw,
    nominal_roll,
    off_nominal_roll,
)
from ..sun import pitch as sun_pitch
from ..sun import position


def test_allowed_rolldev():

    # Test array of pitchs and allowed roll dev
    testarr = [[135, 13.979],
               [138, 14.516],
               [0, 0],
               [40, 0],
               [179.9, 18.748772],
               [179.997, 0],
               [180, 0],
               [181, 0],
               [85.49229, 13.677669],
               [85.52, 18.756727],
               [124.99, 18.748772],
               [125, 17.0]]
    for pitch, rolldev in testarr:
        assert np.isclose(allowed_rolldev(pitch), rolldev)

    # Also test with pitch as vector
    assert np.allclose(allowed_rolldev(np.array(testarr)[:, 0]),
                       np.array(testarr)[:, 1])


def test_position():
    ra, dec = position('2008:002:00:01:02')
    assert np.allclose((ra, dec), (281.903448, -22.989273))


def test_nominal_roll():
    roll = nominal_roll(205.3105, -14.6925, time='2011:019:20:51:13')
    assert np.allclose(roll, 68.83020)  # vs. 68.80 for obsid 12393 in JAN1711A


def test_off_nominal_roll_and_pitch():
    att = (198.392135, 36.594359, 33.983322)  # RA, Dec, Roll of obsid 16354
    oroll = off_nominal_roll(att, '2015:335:00:00:00')  # NOT the 16354 time
    assert np.isclose(oroll, -12.224010)

    date = '2015:077:01:07:04'
    pitch = sun_pitch(att[0], att[1], time=date)
    assert np.isclose(pitch, 139.5651)  # vs. 139.59 in SOT MP page

    pitch, _ = get_sun_pitch_yaw(att[0], att[1], time=date)
    assert np.isclose(pitch, 139.5651)  # vs. 139.59 in SOT MP page


def test_apply_get_sun_pitch_yaw():
    """Test apply and get sun_pitch_yaw with multiple components"""
    att = apply_sun_pitch_yaw([0, 45, 0], pitch=[0, 10, 20], yaw=[0, 5, 10],
                              sun_ra=0, sun_dec=90)
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
    atts = apply_sun_pitch_yaw(att=[0, 45, 10], pitch=dpitches, yaw=dyaws, sun_ra=0, sun_dec=90)
    assert atts.shape == (2, 3)
    exp = np.array(
        [[[355., 45., 10.],
          [360., 45., 10.],
          [5., 45., 10.]],
         [[355., 48., 10.],
          [0., 48., 10.],
          [5., 48., 10.]]])
    assert np.allclose(atts.equatorial, exp)


def test_get_sun_pitch_yaw():
    """Test that values approximately match those from ORviewer.

    See slack discussion "ORviewer sun / anti-sun plots azimuthal Sun yaw angle"
    """
    pitch, yaw = get_sun_pitch_yaw(109, 55.3, time='2021:242')
    assert np.allclose((pitch, yaw), (60.453385, 29.880125))
    pitch, yaw = get_sun_pitch_yaw(238.2, -58.9, time='2021:242')
    assert np.allclose((pitch, yaw), (92.405603, 210.56582))
    pitch, yaw = get_sun_pitch_yaw(338, -9.1, time='2021:242')
    assert np.allclose((pitch, yaw), (179.417797, 259.703451))
