# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from Quaternion import Quat

import ska_sun
from ska_sun import (
    allowed_rolldev,
    apply_sun_pitch_yaw,
    get_sun_pitch_yaw,
    nominal_roll,
    off_nominal_roll,
)
from ska_sun import pitch as sun_pitch
from ska_sun import position

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


def test_allowed_rolldev_with_roll_table(monkeypatch):
    """Test computing scalar values with explicit roll table"""
    monkeypatch.setenv("CHANDRA_MODELS_DEFAULT_VERSION", "3.48")
    roll_table = ska_sun.load_roll_table()
    for pitch, rolldev in exp_pitch_rolldev:
        assert np.isclose(allowed_rolldev(pitch, roll_table), rolldev)


def test_allowed_rolldev_vector_without_roll_table(monkeypatch):
    """Test passing a vector input and NO explicit roll table"""
    monkeypatch.setenv("CHANDRA_MODELS_DEFAULT_VERSION", "3.48")
    assert np.allclose(
        allowed_rolldev(exp_pitch_rolldev[:, 0]), exp_pitch_rolldev[:, 1]
    )


def test_duplicate_pitch_rolldev(monkeypatch):
    # This is a commit of the 2023_020 pitch/roll constraint file that is exactly what
    # was provided by the FOT (except adding the header columns). It contains several
    # duplicate pitch values, including pitch=85.5 with 12.436 and 17.5 roll dev vals.
    # The test is to make sure that the code handles this.
    monkeypatch.setenv("CHANDRA_MODELS_DEFAULT_VERSION", "68670fc")
    assert np.isclose(allowed_rolldev(85.5), 17.5, rtol=0, atol=1e-6)
    assert np.isclose(allowed_rolldev(85.5 - 1e-8), 12.436, rtol=0, atol=1e-6)
    roll_table = ska_sun.load_roll_table()
    pitch_min = roll_table["pitch"][0]
    assert np.isclose(allowed_rolldev(pitch_min - 1e-8), -1.0, rtol=0, atol=1e-6)
    pitch_max = roll_table["pitch"][-1]
    assert np.isclose(allowed_rolldev(pitch_max + 1e-8), -1.0, rtol=0, atol=1e-6)


def test_position():
    ra, dec = position("2008:002:00:01:02")
    assert np.allclose((ra, dec), (281.903448, -22.989273))


def test_position_diff_methods():
    # First confirm that the method works via option or conf
    # Use the time of the beginning of obsid 17198
    time = "2015:338:08:39:46.858"
    ra_fast1, dec_fast1 = position(time, method="fast")
    with ska_sun.conf.set_temp("sun_position_method_default", "fast"):
        ra_fast2, dec_fast2 = position(time)
    assert ra_fast1 == ra_fast2
    assert dec_fast1 == dec_fast2
    # Compare against regression values
    assert np.isclose(ra_fast1, 250.34925)
    assert np.isclose(dec_fast1, -22.20589)

    # Review with accurate method (but with default from_chandra still False)
    ra_acc1, dec_acc1 = position(time, method="accurate")
    with ska_sun.conf.set_temp("sun_position_method_default", "accurate"):
        ra_acc2, dec_acc2 = position(time)
    assert ra_acc1 == ra_acc2
    assert dec_acc1 == dec_acc2
    # Compare against regression values
    assert np.isclose(ra_acc1, 250.11781)
    assert np.isclose(dec_acc1, -22.17919)

    # Show that the spacecraft pitch is a better match for the accurate method
    # Pitch is fetch.Msid('DP_PITCH', '2015:338:08:39:46.858', '2015:338:08:40:46.858').vals[0]
    exp_pitch = 137.05916
    targ_ra = 29.083099133566247
    targ_dec = 5.61232476130315
    sun_pitch_acc = ska_sun.pitch(targ_ra, targ_dec, sun_ra=ra_acc1, sun_dec=dec_acc1)
    sun_pitch_fast = ska_sun.pitch(
        targ_ra, targ_dec, sun_ra=ra_fast1, sun_dec=dec_fast1
    )

    # The accurate method is a close match even without from_chandra=True option to position()
    assert np.isclose(sun_pitch_acc, exp_pitch, rtol=0, atol=1e-3)
    assert not np.isclose(sun_pitch_fast, exp_pitch, rtol=0, atol=1e-2)
    assert np.abs(sun_pitch_acc - exp_pitch) < np.abs(sun_pitch_fast - exp_pitch)

    # And much slower but a little closer match with from_chandra=True
    # which should be the reference giving the exp_pitch anyway
    ra_slow, dec_slow = position(time, method="accurate", from_chandra=True)
    sun_pitch_slow = ska_sun.pitch(targ_ra, targ_dec, sun_ra=ra_slow, sun_dec=dec_slow)
    assert np.isclose(sun_pitch_slow, exp_pitch, rtol=0, atol=2e-5)


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
    # A sampling of args from the roll table meta
    roll_table = ska_sun.load_roll_table()
    exp = {
        "file_path": "chandra_models/pitch_roll/pitch_roll_constraint.csv",
        "version": None,
        "repo_path": "None",
        "require_latest_version": False,
        "timeout": 5,
    }
    for key, val in exp.items():
        assert roll_table.meta["call_args"][key] == val


def test_roll_table_pitch_increasing():
    """Check that the pitch values are monotonically increasing. Duplicate values
    are allowed and np.interp will choose the second one in this case.
    """
    dat = ska_sun.load_roll_table()
    assert np.all(np.diff(dat["pitch"]) >= 0)
