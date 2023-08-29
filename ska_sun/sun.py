# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utility for calculating sun position, pitch angle and values related to roll.
"""
from math import acos, asin, atan2, cos, degrees, pi, radians, sin

import numba
import numpy as np
from astropy.table import Table
from Chandra.Time import DateTime
from chandra_aca.planets import get_planet_chandra, get_planet_eci
from chandra_aca.transform import eci_to_radec, radec_to_eci
from Quaternion import Quat
from ska_helpers import chandra_models

CHANDRA_MODELS_PITCH_ROLL_FILE = "chandra_models/pitch_roll/pitch_roll_constraint.csv"

from . import conf

__all__ = [
    "allowed_rolldev",
    "position",
    "position_fast",
    "position_accurate",
    "sph_dist",
    "pitch",
    "load_roll_table",
    "nominal_roll",
    "off_nominal_roll",
    "get_sun_pitch_yaw",
    "apply_sun_pitch_yaw",
]


def _roll_table_read_func(filename):
    return Table.read(filename), filename


@chandra_models.chandra_models_cache
def load_roll_table():
    """Load the pitch/roll table from the chandra_models repo.

    The result depends on environment variables:

    - ``CHANDRA_MODELS_REPO_DIR``: root directory of chandra_models repo
    - ``CHANDRA_MODELS_DEFAULT_VERSION``: default version of chandra_models to use

    :returns: ``astropy.table.Table``
        Table with "pitch" and "off_nom_roll" columns. Detailed provenance information
        is available in the table ``meta`` attribute.
    """
    dat, info = chandra_models.get_data(
        CHANDRA_MODELS_PITCH_ROLL_FILE, read_func=_roll_table_read_func
    )
    dat.meta.update(info)

    return dat


def allowed_rolldev(pitch, roll_table=None):
    """Get allowed roll deviation (off-nominal roll) for the given ``pitch``.

    This performs a linear interpolation of the values in the pitch/roll table in
    the chandra_models repo in ``chandra_models/pitch_roll/pitch_roll_constraint.csv``.

    For pitch values outside the range of the table the returned rolldev is -1.0,
    corresonding to a pitch angle outside of the planning limits.

    :param pitch: float, ndarray
        Sun pitch angle (deg)
    :param roll_table: astropy.table.Table
        Table of pitch/roll values (optional)
    :returns: float, ndarray
        Roll deviation (deg)
    """
    if roll_table is None:
        roll_table = load_roll_table()

    out = np.interp(
        x=pitch,
        xp=roll_table["pitch"],
        fp=roll_table["off_nom_roll"],
        left=-1.0,
        right=-1.0,
    )
    return out


# The position() method is a modification of
# http://idlastro.gsfc.nasa.gov/ftp/pro/astro/sunpos.pro
#
#  Copyright (c) 2014, Wayne Landsman
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#
#      Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#      Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in
#      the documentation and/or other materials provided with the
#      distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


def position_fast(time):
    """
    Calculate the sun position at the given ``time`` using a fast approximation.

    Code modified from http://idlastro.gsfc.nasa.gov/ftp/pro/astro/sunpos.pro

    This implementation is returns coordinates that are in error by as much as
    0.3 deg. Use the ``position_accurate()`` function or ``position(.., method='accurate')``
    unless speed is critical.

    This function is about 40x faster than the ``position_accurate()`` function (30 us for a
    single time vs 1.2 ms). However, the ``position_accurate()`` function can be vectorized and
    the speed difference is reduced.

    Example::

     >>> import ska_sun
     >>> ska_sun.position_fast('2008:002:00:01:02')
     (281.90344855695275, -22.9892737322084)

    :param time: Input time (Chandra.Time compatible format)
    :rtype: RA, Dec in decimal degrees (J2000).
    """
    # Most of the computational time is spent converting to JD.
    time_jd = DateTime(time).jd
    out = position_at_jd(time_jd)
    return out


@numba.njit(cache=True)
def position_at_jd(jd):
    t = (jd - 2415020) / (36525.0)

    dtor = pi / 180

    # sun's mean longitude
    lon = (279.696678 + ((36000.768925 * t) % 360.0)) * 3600.0

    # Earth anomaly
    me = 358.475844 + (35999.049750 * t) % 360.0
    ellcor = (6910.1 - (17.2 * t)) * sin(me * dtor) + 72.3 * sin(2.0 * me * dtor)
    lon = lon + ellcor

    # allow for the Venus perturbations using the mean anomaly of Venus MV
    mv = 212.603219 + (58517.803875 * t) % 360.0
    vencorr = (
        4.8 * cos((299.1017 + mv - me) * dtor)
        + 5.5 * cos((148.3133 + 2.0 * mv - 2.0 * me) * dtor)
        + 2.5 * cos((315.9433 + 2.0 * mv - 3.0 * me) * dtor)
        + 1.6 * cos((345.2533 + 3.0 * mv - 4.0 * me) * dtor)
        + 1.0 * cos((318.15 + 3.0 * mv - 5.0 * me) * dtor)
    )
    lon = lon + vencorr

    #  Allow for the Mars perturbations using the mean anomaly of Mars MM
    mm = 319.529425 + (19139.858500 * t) % 360.0
    marscorr = 2.0 * cos((343.8883 - 2.0 * mm + 2.0 * me) * dtor) + 1.8 * cos(
        (200.4017 - 2.0 * mm + me) * dtor
    )
    lon = lon + marscorr

    # Allow for the Jupiter perturbations using the mean anomaly of Jupiter MJ
    mj = 225.328328 + (3034.6920239 * t) % 360.0
    jupcorr = (
        7.2 * cos((179.5317 - mj + me) * dtor)
        + 2.6 * cos((263.2167 - mj) * dtor)
        + 2.7 * cos((87.1450 - 2.0 * mj + 2.0 * me) * dtor)
        + 1.6 * cos((109.4933 - 2.0 * mj + me) * dtor)
    )
    lon = lon + jupcorr

    # Allow for the Moons perturbations using the mean elongation of the Moon
    # from the Sun D
    d = 350.7376814 + (445267.11422 * t) % 360.0
    mooncorr = 6.5 * sin(d * dtor)
    lon = lon + mooncorr

    # Allow for long period terms
    longterm = 6.4 * sin((231.19 + 20.20 * t) * dtor)
    lon = lon + longterm
    lon = (lon + 2592000.0) % 1296000.0

    # Allow for Aberration
    lon = lon - 20.5

    # Allow for Nutation using the longitude of the Moons mean node OMEGA
    omega = 259.183275 - (1934.142008 * t) % 360.0
    lon = lon - 17.2 * sin(omega * dtor)

    # Form the True Obliquity
    oblt = 23.452294 - 0.0130125 * t + (9.2 * cos(omega * dtor)) / 3600.0

    # Form Right Ascension and Declination
    lon = lon / 3600.0
    ra = atan2(sin(lon * dtor) * cos(oblt * dtor), cos(lon * dtor))

    while (ra < 0) or (ra > (2 * pi)):
        if ra < 0:
            ra += 2 * pi
        if ra > (2 * pi):
            ra -= 2 * pi

    dec = asin(sin(lon * dtor) * sin(oblt * dtor))

    return ra / dtor, dec / dtor


def position_accurate(time, from_chandra=False):
    """
    Calculate the sun RA, Dec at the given ``time`` from Earth geocenter or Chandra.

    By default the position is calculated from Earth geocenter.  If ``from_chandra=True``
    the position is calculated from Chandra using the Chandra predictive ephemeris
    via the cheta telemetry archive.

    These methods rely on the DE432 ephemeris and functions in ``chandra_aca.planets``.
    With ``from_chandra=True`` the position should be accurate to within a few arcsec.
    With ``from_chandra=False`` the position is accurate to within about 0.05 deg.

    Example::

     >>> import ska_sun
     >>> ska_sun.position_accurate('2008:002:00:01:02')
     (281.7865848220755, -22.99607130644057)

    :param time: Input time(s) (CxoTimeLike)
    :param from_chandra: If True compute position from Chandra using ephemeris in cheta
    :rtype: RA, Dec in decimal degrees (J2000).
    """
    func = get_planet_chandra if from_chandra else get_planet_eci
    eci_sun = func("sun", time)
    ra, dec = eci_to_radec(eci_sun)
    return ra, dec


def position(time, method=None, **kwargs):
    """
    Calculate the sun RA, Dec at the given ``time`` from Earth geocenter or Chandra.

    The method for position determination may be explicitly set via kwarg to ``fast`` or
    ``accurate``.  See `position_fast()` and `position_accurate` methods for details.
    The default method behavior is set by ``ska_sun.conf.sun_position_method_default``,
    currently set to `fast`.

    The ``accurate`` method also supports the ``from_chandra`` kwarg.

    Example::

     >>> import ska_sun
     >>> ska_sun.position('2008:002:00:01:02')
     (281.90344855695275, -22.9892737322084)
     >>> ska_sun.position('2008:002:00:01:02', method='accurate')
     (281.7865848220755, -22.99607130644057)
     >>> with ska_sun.conf.set_temp('sun_position_method_default', 'accurate'):
     ...    ska_sun.position('2008:002:00:01:02')
     (281.7865848220755, -22.99607130644057
     >>> ska_sun.position('2008:002:00:01:02', method='accurate', from_chandra=True)
     (281.80963749492935, -23.033877980418676)

    :param time: Input time(s) (CxoTimeLike)
    :param method: Method to use ("fast" | "accurate", default="fast")
    :param **kwargs: Passed to position_<method>()
    :rtype: RA, Dec in decimal degrees (J2000)
    """
    if method is None:
        method = conf.sun_position_method_default

    try:
        position_func = globals()["position_" + method]
    except KeyError:
        raise ValueError(f"Invalid sun position method '{method}'") from None

    ra, dec = position_func(time, **kwargs)
    return ra, dec


@numba.njit(cache=True)
def sph_dist(a1, d1, a2, d2):
    """Calculate spherical distance between two sky positions.

    Not highly accurate for very small angles.  This function is deprecated, use
    agasc.sphere_dist() instead.

    :param a1: RA position 1 (deg)
    :param d1: dec position 1 (deg)
    :param a2: RA position 2 (deg)
    :param d2: dec position 2 (deg)

    :rtype: spherical distance (deg)
    """
    if a1 == a2 and d1 == d2:
        return 0.0
    a1 = radians(a1)
    d1 = radians(d1)
    a2 = radians(a2)
    d2 = radians(d2)
    val = cos(d1) * cos(d2) * cos(a1 - a2) + sin(d1) * sin(d2)
    if val > 1.0:
        val = 1.0
    elif val < -1.0:
        val = -1.0
    return degrees(acos(val))


def pitch(ra, dec, time=None, sun_ra=None, sun_dec=None):
    """Calculate sun pitch angle for spacecraft attitude ``ra``, ``dec``.

    You can provide either ``time`` or explicit values of ``sun_ra`` and ``sun_dec``.

    Example::

     >>> ska_sun.pitch(10., 20., '2009:001:00:01:02')
     96.256434327840864

    :param ra: target right ascension (deg)
    :param dec: target declination (deg)
    :param time: time (CxoTimeLike) [optional]
    :param sun_ra: sun RA (deg) instead of time [optional]
    :param sun_dec: sun Dec (deg) instead of time [optional]

    :returns: sun pitch angle (deg)
    """
    if time is not None:
        sun_ra, sun_dec = position(time)
    pitch = sph_dist(ra, dec, sun_ra, sun_dec)

    return pitch


@numba.njit(cache=True)
def _radec2eci(ra, dec):
    """
    Convert from RA,Dec to ECI for single RA,Dec pair.

    This is a numba-ized version of the original code in Ska.quatutil.

    :param ra: Right Ascension (float, degrees)
    :param dec: Declination (float, degrees)
    :returns: numpy array ECI (3-vector)
    """
    r = np.radians(ra)
    d = np.radians(dec)
    return np.array([np.cos(r) * np.cos(d), np.sin(r) * np.cos(d), np.sin(d)])


def nominal_roll(ra, dec, time=None, sun_ra=None, sun_dec=None):
    """Calculate nominal roll angle for the given spacecraft attitude ``ra``,
    ``dec`` at ``time``.  Optionally one can provide explicit values of
    ``sun_ra`` and ``sun_dec`` instead of ``time``.

    Example::

      >>> ska_sun.nominal_roll(205.3105, -14.6925, time='2011:019:20:51:13')
      68.830209134280665    # vs. 68.80 for obsid 12393 in JAN1711A

    :param ra: right ascension
    :param dec: declination
    :param time: time (any Chandra.Time format) [optional]
    :param sun_ra: Sun right ascension (instead of ``time``)
    :param sun_dec: Sun declination (instead of ``time``)

    :returns: nominal roll angle (deg)

    """
    if time is not None:
        sun_ra, sun_dec = position(time)
    roll = _nominal_roll(ra, dec, sun_ra, sun_dec)
    return roll


@numba.njit(cache=True)
def _nominal_roll(ra, dec, sun_ra, sun_dec):
    sun_eci = _radec2eci(sun_ra, sun_dec)
    body_x = _radec2eci(ra, dec)
    if np.sum((sun_eci - body_x) ** 2) < 1e-10:
        raise ValueError("No nominal roll for ra, dec == sun_ra, sun_dec")
    body_y = np.cross(body_x, sun_eci)
    body_y = body_y / np.sqrt(np.sum(body_y**2))
    body_z = np.cross(body_x, body_y)
    body_z = body_z / np.sqrt(
        np.sum(body_z**2)
    )  # shouldn't be needed but do it anyway
    roll = np.degrees(np.arctan2(body_y[2], body_z[2]))
    return roll


def off_nominal_roll(att, time=None, sun_ra=None, sun_dec=None):
    """
    Calculate off-nominal roll angle for spacecraft attitude ``att``.

    This function is not vectorized so inputs must be scalars.

    Off-nominal roll is defined as roll - nominal roll. If ``time`` is provided
    then the sun position is calculated from ``time``, otherwise you must
    provide ``sun_ra`` and ``sun_dec``.

    Example::

      >>> att = (198.392135, 36.594359, 33.983322)  # RA, Dec, Roll of obsid 16354
      >>> ska_sun.off_nominal_roll(att, '2015:335:00:00:00')
      -12.22401097980562

    :param att: any Quat() value (e.g. [ra, dec, roll] or [q1, q2, q3, q4])
    :param time: any DateTime value
    :param sun_ra: sun RA (deg) instead of time [optional]
    :param sun_dec: sun Dec (deg) instead of time [optional]

    :returns: off nominal roll angle (deg)
    """
    if time is not None:
        sun_ra, sun_dec = position(time)

    if isinstance(att, Quat):
        ra, dec, roll = att.equatorial
    elif len(att) == 3:
        ra, dec, roll = att
    else:
        q = Quat(att)
        ra, dec, roll = q.equatorial

    nom_roll = _nominal_roll(ra, dec, sun_ra, sun_dec)
    off_nom_roll = roll - nom_roll

    if off_nom_roll < -180:
        off_nom_roll += 360
    elif off_nom_roll >= 180:
        off_nom_roll -= 360

    return off_nom_roll


def get_sun_pitch_yaw(ra, dec, time=None, sun_ra=None, sun_dec=None):
    """Get Sun pitch and yaw angles of Sky coordinate(s).

    :param ra: float, ndarray
        RA(s)
    :param dec: float, ndarray
        Dec(s)
    :param time: date-like, optional
        Date of observation.  If not given, use ``sun_ra`` and ``sun_dec``
        if provided or else use current date.
    :param sun_ra: float, optional
        RA of sun.  If not given, use estimated sun RA at ``date``.
    :param sun_dec: float, optional
        Dec of sun.  If not given, use estimated sun dec at ``date``.

    :returns:
        2-tuple (pitch, yaw) in degrees.
    """
    # If not provided calculate sun RA and Dec using a low-accuracy ephemeris
    if sun_ra is None or sun_dec is None:
        sun_ra, sun_dec = position(time)

    # Compute attitude vector in ECI
    att_eci = radec_to_eci(ra, dec)

    # Make a Sun frame defined by vector to the Sun assuming roll=0
    sun_frame = Quat([sun_ra, sun_dec, 0])
    # Sun frame inverse rotation matrix.
    sun_frame_rot = sun_frame.transform.T

    # Compute attitude vector in Sun frame.
    att_sun = np.einsum("...jk,...k->...j", sun_frame_rot, att_eci)

    # Usual for pitch and yaw. The yaw is set to match ORviewer:
    # get_sun_pitch_yaw(109, 55.3, time='2021:242') ~ (60, 30)
    # get_sun_pitch_yaw(238.2, -58.9, time='2021:242') ~ (90, 210)
    pitch = np.arccos(att_sun[..., 0])
    yaw = -np.arctan2(att_sun[..., 1], att_sun[..., 2])  # -pi <= yaw < pi
    yaw = yaw % (2 * np.pi)  # 0 <= yaw < 2pi

    return np.rad2deg(pitch), np.rad2deg(yaw)


def apply_sun_pitch_yaw(att, pitch=0, yaw=0, time=None, sun_ra=None, sun_dec=None):
    """Apply pitch(es) and yaw(s) about Sun line to an attitude.

    :param att: Quaternion-like
        Attitude(s) to be rotated.
    :param pitch: float, ndarray
        Sun pitch offsets (deg)
    :param yaw: float, ndarray
        Sun yaw offsets (deg)
    :param sun_ra: float, optional
        RA of sun.  If not given, use estimated sun RA at ``time``.
    :param sun_dec: float, optional
        Dec of sun.  If not given, use estimated sun dec at ``time``.

    :returns: Quat
        Modified attitude(s)

    """
    if not isinstance(att, Quat):
        att = Quat(att)

    # If not provided calculate sun RA and Dec using a low-accuracy ephemeris
    if sun_ra is None or sun_dec is None:
        sun_ra, sun_dec = position(time)

    # Compute Sun and attitude vectors in ECI
    eci_sun = radec_to_eci(sun_ra, sun_dec)
    eci_att = radec_to_eci(att.ra, att.dec)

    # Rotation vector for apply pitch about Sun line
    pitch_rot_vec = np.cross(eci_sun, eci_att)
    pitch_rot_vec = pitch_rot_vec / np.linalg.norm(pitch_rot_vec)

    # Broadcast input pitch and yaw to a common shape
    pitches, yaws = np.broadcast_arrays(pitch, yaw)
    out_shape = pitches.shape
    # Get pitches and yaws as 1-d iterables
    pitches = np.atleast_1d(pitches).ravel()
    yaws = np.atleast_1d(yaws).ravel()

    qs = []  # Output quaternions as a list of 4-element ndarrays
    for pitch, yaw in zip(pitches, yaws):
        att_out = att
        if pitch != 0:
            # Pitch rotation is in the plane defined by attitude vector and the
            # body-to-sun vector.
            att_out = att_out.rotate_about_vec(pitch_rot_vec, pitch)
        if yaw != 0:
            # Yaw rotation is about the body-to-sun vector.
            att_out = att_out.rotate_about_vec(eci_sun, yaw)
        qs.append(att_out.q)

    # Reshape into correct output shape and return corresponding quaternion
    qs = np.array(qs).reshape(out_shape + (4,))
    return Quat(q=qs)
