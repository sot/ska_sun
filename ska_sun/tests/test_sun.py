# Licensed under a 3-clause BSD style license - see LICENSE.rst

import astropy.units as u
import numpy as np
import pytest
from cxotime import CxoTime
from Quaternion import Quat

import ska_sun
from ska_sun import (
    allowed_rolldev,
    apply_sun_pitch_yaw,
    get_nsm_attitude,
    get_sun_pitch_yaw,
    nominal_roll,
    off_nominal_roll,
)
from ska_sun import pitch as sun_pitch
from ska_sun import position


@pytest.fixture()
def fast_sun_position_method(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ska_sun.conf, "sun_position_method_default", "fast")


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


def test_position(fast_sun_position_method):
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
    # Good within 3 arcsec
    assert np.isclose(sun_pitch_slow, exp_pitch, rtol=0, atol=3 / 3600)


def test_nominal_roll(fast_sun_position_method):
    roll = nominal_roll(205.3105, -14.6925, time="2011:019:20:51:13")
    assert np.allclose(roll, 68.83020)  # vs. 68.80 for obsid 12393 in JAN1711A


def test_nominal_roll_range(fast_sun_position_method):
    roll = nominal_roll(0, 89.9, time="2019:006:12:00:00")
    assert np.allclose(roll, 287.24879)  # range in 0-360 and value for sparkles test


def test_off_nominal_roll_and_pitch(fast_sun_position_method):
    att = (198.392135, 36.594359, 33.983322)  # RA, Dec, Roll of obsid 16354
    oroll = off_nominal_roll(att, "2015:335:00:00:00")  # NOT the 16354 time
    assert np.isclose(oroll, -12.224010)

    date = "2015:077:01:07:04"
    pitch = sun_pitch(att[0], att[1], time=date)
    assert np.isclose(pitch, 139.5651)  # vs. 139.59 in SOT MP page

    pitch, _ = get_sun_pitch_yaw(att[0], att[1], time=date)
    assert np.isclose(pitch, 139.5651)  # vs. 139.59 in SOT MP page


@pytest.mark.parametrize("coord_system", ["spacecraft", "ORviewer"])
def test_apply_get_sun_pitch_yaw(coord_system):
    """Test apply and get sun_pitch_yaw with multiple components"""
    att = apply_sun_pitch_yaw(
        [0, 45, 0],
        pitch=[0, 10, 20],
        yaw=[0, 5, 10],
        sun_ra=0,
        sun_dec=90,
        coord_system=coord_system,
    )
    pitch, yaw = get_sun_pitch_yaw(
        att.ra, att.dec, sun_ra=0, sun_dec=90, coord_system=coord_system
    )
    # The results are the same for both coord_systems because apply_ and get_ have the
    # same sign convention for yaw.
    assert np.allclose(pitch, 45 + np.array([0, 10, 20]))
    assert np.allclose(yaw, 180 + np.array([0, 5, 10]))


@pytest.mark.parametrize("coord_system", ["spacecraft", "ORviewer"])
def test_apply_sun_pitch_yaw(coord_system):
    """Basic test of apply_sun_pitch_yaw"""
    att = Quat(equatorial=[0, 45, 0])
    att2 = apply_sun_pitch_yaw(
        att, pitch=10, yaw=0, sun_ra=0, sun_dec=0, coord_system=coord_system
    )
    assert np.allclose((att2.ra, att2.dec), (0, 55))

    att2 = apply_sun_pitch_yaw(
        att, pitch=0, yaw=10, sun_ra=0, sun_dec=90, coord_system=coord_system
    )
    # In this case we apply a yaw and so the result depends on the coord_system.
    sign = -1.0 if coord_system == "spacecraft" else 1.0
    assert np.allclose((att2.ra, att2.dec), ((sign * 10) % 360, 45))


def test_apply_sun_pitch_yaw_with_grid():
    """Use np.ogrid to make a grid of RA/Dec values (via dpitches and dyaws)"""
    dpitches, dyaws = np.ogrid[0:-3:2j, -5:5:3j]
    atts = apply_sun_pitch_yaw(
        att=[1, 45, 10],
        pitch=dpitches,
        yaw=dyaws,
        sun_ra=0,
        sun_dec=90,
        coord_system="ORviewer",
    )
    assert atts.shape == (2, 3)
    exp = np.array(
        [
            [[356.0, 45.0, 10.0], [1.0, 45.0, 10.0], [6.0, 45.0, 10.0]],
            [[356.0, 48.0, 10.0], [1.0, 48.0, 10.0], [6.0, 48.0, 10.0]],
        ]
    )
    assert np.allclose(atts.equatorial, exp)


def test_get_sun_pitch_yaw(fast_sun_position_method):
    """Test that values approximately match those from ORviewer.

    See slack discussion "ORviewer sun / anti-sun plots azimuthal Sun yaw angle"
    """
    pitch, yaw = get_sun_pitch_yaw(109, 55.3, time="2021:242", coord_system="ORviewer")
    assert np.allclose((pitch, yaw), (60.453385, 29.880125))
    pitch, yaw = get_sun_pitch_yaw(
        238.2, -58.9, time="2021:242", coord_system="ORviewer"
    )
    assert np.allclose((pitch, yaw), (92.405603, 210.56582))
    pitch, yaw = get_sun_pitch_yaw(338, -9.1, time="2021:242", coord_system="ORviewer")
    assert np.allclose((pitch, yaw), (179.417797, 259.703451))


def test_apply_get_sun_pitch_yaw_spacecraft_sign():
    """Test the sign of a yaw applied to a spacecraft attitude.

    This uses real telemetry from the 2024:036 NSM recovery activity. At around
    2024:036:01:32:00.000, a "positive yaw bias of 0.025 deg/s for 30 minutes" was
    applied. The yaw maneuver converged to a stable attitude around  0210z. The
    expectation is that pitch is 90.0 and yaw *increases* by 45 degrees of the maneuver.
    """
    # From telemetry at 2024:036 0130z and 0210z
    att0 = Quat([60.35823393, -34.92322161, 291.13817936])
    att1 = Quat([111.94789334, -71.01495527, 335.27170577])
    pitch0, yaw0 = get_sun_pitch_yaw(att0.ra, att0.dec, time="2024:036:01:30:00")
    pitch1, yaw1 = get_sun_pitch_yaw(att1.ra, att1.dec, time="2024:036:02:10:00")
    assert np.isclose(pitch0, 90.0, atol=1)
    assert np.isclose(pitch1, 90.0, atol=1)
    assert np.isclose(yaw1 - yaw0, 45, atol=1)

    # Apply yaw bias to the 0130z attitude using sun position midway. This gives
    # ra, dec, roll = [111.44329433 -71.34681456 335.48326522].
    att1_apply = apply_sun_pitch_yaw(att0, pitch=0, yaw=45, time="2024:036:01:47:00")
    dq = att1_apply.dq(att1)
    assert abs(dq.pitch) < 1
    assert abs(dq.yaw) < 1
    assert abs(dq.roll0) < 1


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


@pytest.mark.parametrize("method", ["fast", "accurate"])
def test_array_input_and_different_formats(method):
    date0 = CxoTime("2019:001")
    dates = np.array([date0 + i_dt * u.day for i_dt in np.arange(0, 10, 1)])
    times = [date.secs for date in dates]
    # Make sure list, array, CxoTime array inputs work
    pos1 = ska_sun.position(times, method=method)
    pos2 = ska_sun.position(dates, method=method)
    pos3 = ska_sun.position(CxoTime(dates), method=method)
    for idx in range(2):
        assert np.all(pos1[idx] == pos2[idx])
        assert np.all(pos1[idx] == pos3[idx])

    for ra, dec, time in zip(pos1[0], pos1[1], times):
        ra2, dec2 = ska_sun.position(time, method=method)
        assert np.isclose(ra, ra2, rtol=0, atol=1e-13)
        assert np.isclose(dec, dec2, rtol=0, atol=1e-13)


@pytest.mark.parametrize("pitch", [90, 130, 160])
def test_nsm_attitude_random_atts(pitch):
    np.random.seed(0)
    n_test = 10
    date = "2024:001"
    ras = np.random.uniform(0, 360, n_test)
    decs = np.random.uniform(-90, 90, n_test)
    rolls = [
        ska_sun.nominal_roll(ra, dec, time=date) + np.random.uniform(-20, 20)
        for ra, dec in zip(ras, decs)
    ]

    for ra, dec, roll in zip(ras, decs, rolls):
        att0 = Quat([ra, dec, roll])
        att_nsm = get_nsm_attitude(att0, date, pitch)
        pitch_nsm = ska_sun.pitch(att_nsm.ra, att_nsm.dec, date)
        assert np.isclose(pitch_nsm, pitch, rtol=0, atol=1e-4)
        off_nom_roll_nsm = ska_sun.off_nominal_roll(att_nsm, date)
        assert np.isclose(off_nom_roll_nsm, 0, rtol=0, atol=1e-4)


def test_nsm_attitude_corner_case():
    """Attitude which is already at exactly the NSM attitude.
    This tests the lines::
      if norm2 < 1e-16:
          rot_axis = np.array([1., 0., 0.])
    In development, a temporary print statement verified that this branch was taken.
    Attitude is from::
      ska_sun.get_att_for_sun_pitch_yaw(time="2024:001", pitch=90, yaw=85, off_nom_roll=0)
    """
    date = "2024:001"
    att0 = Quat(
        [
            -0.5462715389868936,
            -0.07466130207138927,
            0.04050165010577328,
            0.8332902927579349,
        ]
    )
    att_nsm = get_nsm_attitude(att0, "2024:001")
    pitch = ska_sun.pitch(att0.ra, att0.dec, date)
    pitch_nsm = ska_sun.pitch(att_nsm.ra, att_nsm.dec, date)
    assert np.isclose(pitch, 90, rtol=0, atol=1e-4)
    assert np.isclose(pitch_nsm, 90, rtol=0, atol=1e-4)
    off_nom_roll0 = ska_sun.off_nominal_roll(att0, date)
    off_nom_roll_nsm = ska_sun.off_nominal_roll(att_nsm, date)
    assert np.isclose(off_nom_roll0, 0, rtol=0, atol=1e-4)
    assert np.isclose(off_nom_roll_nsm, 0, rtol=0, atol=1e-4)


@pytest.mark.parametrize("use_time", [True, False])
@pytest.mark.parametrize("coord_system", ["spacecraft", "ORviewer", None])
def test_get_att_for_sun_pitch_yaw(use_time: bool, coord_system):
    np.random.seed(1)
    n_test = 100
    dates = CxoTime("2024:001") + np.random.uniform(0, 1, n_test) * u.yr
    pitches = np.random.uniform(45, 178.0, n_test)
    yaws = np.random.uniform(0, 360, n_test)
    off_nom_rolls = np.random.uniform(-20, 20, n_test)
    for date, pitch, yaw, off_nom_roll in zip(dates, pitches, yaws, off_nom_rolls):
        sun_ra, sun_dec = ska_sun.position(date)
        kwargs_pos = (
            {"time": date} if use_time else {"sun_ra": sun_ra, "sun_dec": sun_dec}
        )
        kwargs_coord = {} if coord_system is None else {"coord_system": coord_system}
        att = ska_sun.get_att_for_sun_pitch_yaw(
            pitch, yaw, off_nom_roll=off_nom_roll, **(kwargs_pos | kwargs_coord)
        )
        pitch_out, yaw_out = ska_sun.get_sun_pitch_yaw(
            att.ra, att.dec, **(kwargs_pos | kwargs_coord)
        )
        off_nom_roll_out = ska_sun.off_nominal_roll(att, **kwargs_pos)
        assert np.isclose(pitch_out, pitch, rtol=0, atol=1e-4)
        assert np.isclose(yaw_out, yaw, rtol=0, atol=1e-4)
        assert np.isclose(off_nom_roll_out, off_nom_roll, rtol=0, atol=1e-4)
