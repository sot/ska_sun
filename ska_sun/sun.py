# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utility for calculating sun position, pitch angle and values related to roll.
"""

import numba
import numpy as np
from astropy.table import Table
from chandra_aca.planets import get_planet_chandra, get_planet_eci
from chandra_aca.transform import eci_to_radec, radec_to_eci
from cxotime import CxoTimeLike, convert_time_format
from numpy import arccos as acos
from numpy import arcsin as asin
from numpy import arctan2 as atan2
from numpy import cos, degrees, pi, radians, sin
from Quaternion import Quat, QuatLike
from ska_helpers import chandra_models

from ska_sun import conf

CHANDRA_MODELS_PITCH_ROLL_FILE = "chandra_models/pitch_roll/pitch_roll_constraint.csv"

__all__ = [
    "allowed_rolldev",
    "position",
    "position_fast",
    "position_fast_at_jd",
    "position_accurate",
    "sph_dist",
    "pitch",
    "load_roll_table",
    "nominal_roll",
    "off_nominal_roll",
    "get_sun_pitch_yaw",
    "apply_sun_pitch_yaw",
    "get_att_for_sun_pitch_yaw",
    "get_nsm_attitude",
]


def _roll_table_read_func(filename):
    return Table.read(filename), filename


@chandra_models.chandra_models_cache
def load_roll_table():
    """Load the pitch/roll table from the chandra_models repo.

    The result depends on environment variables:

    - ``CHANDRA_MODELS_REPO_DIR``: root directory of chandra_models repo
    - ``CHANDRA_MODELS_DEFAULT_VERSION``: default version of chandra_models to use

    Returns
    -------
    astropy.table.Table
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

    Parameters
    ----------
    pitch : float, ndarray
        Sun pitch angle (deg)
    roll_table : astropy.table.Table
        Table of pitch/roll values (optional)

    Returns
    -------
    float, ndarray
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

    Examples
    --------
    >>> import ska_sun
    >>> ska_sun.position_fast('2008:002:00:01:02')
    (281.90344855695275, -22.9892737322084)

    Parameters
    ----------
    time : CxoTimeLike (scalar)
        Input time.

    Returns
    -------
    sun_ra : float
        Right Ascension in decimal degrees (J2000).
    sun_dec : float
        Declination in decimal degrees (J2000).
    """
    # Most of the computational time is spent converting to JD.
    time_jd = convert_time_format(time, "jd")
    out = position_fast_at_jd(time_jd)
    return out


@numba.njit(cache=True)
def position_fast_at_jd(jd):
    """Calculate the sun position at the given ``time`` (JD) using a fast approximation.

    See ``position_fast()`` for details.

    Parameters
    ----------
    jd : float
        Input time in JD.

    Returns
    -------
    sun_ra : float
        Right Ascension in decimal degrees (J2000).
    sun_dec : float
        Declination in decimal degrees (J2000).
    """
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
    ra = np.mod(ra, 2 * pi)

    dec = asin(sin(lon * dtor) * sin(oblt * dtor))

    return ra / dtor, dec / dtor


def position_accurate(time, from_chandra=False):
    """
    Calculate the sun RA, Dec at the given ``time`` from Earth geocenter or Chandra.

    If ``from_chandra=False`` (default) the position is calculated from Earth geocenter.
    If ``from_chandra=True`` the position is calculated from Chandra using the Chandra
    predictive ephemeris via the cheta telemetry archive.

    These methods rely on the DE432 ephemeris and functions in ``chandra_aca.planets``.
    With ``from_chandra=True`` the position should be accurate to within a few arcsec.
    With ``from_chandra=False`` the position is accurate to within about 0.05 deg.

    Examples
    --------
    >>> import ska_sun
    >>> ska_sun.position_accurate('2008:002:00:01:02')
    (281.7865848220755, -22.99607130644057)

    Parameters
    ----------
    time : CxoTimeLike (scalar)
        Input time.
    from_chandra : bool, optional
        If True compute position from Chandra using ephemeris in cheta.

    Returns
    -------
    sun_ra : float
        Right Ascension in decimal degrees (J2000).
    sun_dec : float
        Declination in decimal degrees (J2000).
    """
    func = get_planet_chandra if from_chandra else get_planet_eci
    eci_sun = func("sun", time)
    ra, dec = eci_to_radec(eci_sun)
    return ra, dec


def position(time, method=None, **kwargs):
    """
    Calculate the sun RA, Dec at the given ``time`` from Earth geocenter or Chandra.

    ``method`` sets the method for computing the sun position which is used for pitch.
    The default is set by ``ska_sun.conf.sun_pitch_roll_method_default``, which defaults
    to ``accurate``.  The available options are:

    - ``accurate``: Use the accurate method (see ``position_accurate()``).
    - ``fast``: Use the fast method (see ``position_fast()``).

    Examples
    --------
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

    Parameters
    ----------
    time : CxoTimeLike (scalar)
        Input time.
    method : str, optional
        Method to use for computing the sun position (see above).
    **kwargs : dict, optional
        Additional keyword arguments passed to the position method.

    Returns
    -------
    sun_ra : float
        Right Ascension in decimal degrees (J2000).
    sun_dec : float
        Declination in decimal degrees (J2000).
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

    Not highly accurate for very small angles, but this function is very fast for
    scalars (about 300 ns on modern hardware).

    Parameters
    ----------
    a1 : float
        RA position 1 (deg)
    d1 : float
        Dec position 1 (deg)
    a2 : float
        RA position 2 (deg)
    d2 : float
        Dec position 2 (deg)

    Returns
    -------
    float
        Spherical distance (deg)
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


def pitch(ra, dec, time=None, sun_ra=None, sun_dec=None, method=None):
    """
    Calculate sun pitch angle for spacecraft attitude.

    If both ``sun_ra`` and ``sun_dec`` are provided then those are used for the Sun
    position; otherwise ``time`` is used to compute the Sun position.

    ``method`` sets the method for computing the sun position which is used for pitch.
    See ``position()`` for details.

    Parameters
    ----------
    ra : float
        Target right ascension (deg).
    dec : float
        Target declination (deg).
    time : CxoTimeLike, optional
        Time for sun position.
    sun_ra : float, optional
        Sun RA (deg) instead of time.
    sun_dec : float, optional
        Sun Dec (deg) instead of time.
    method : str, optional.
        Method for calculating sun position. Valid options are "accurate", "fast".

    Returns
    -------
    float
        Sun pitch angle (deg).

    Examples
    --------
    >>> ska_sun.pitch(10., 20., '2009:001:00:01:02')
    96.256434327840864
    """
    if sun_ra is None or sun_dec is None:
        sun_ra, sun_dec = position(time, method=method)
    pitch = sph_dist(ra, dec, sun_ra, sun_dec)

    return pitch


@numba.njit(cache=True)
def _radec2eci(ra, dec):
    """
    Convert from RA,Dec to ECI for single RA,Dec pair.

    This is a numba-ized version of the original code in Ska.quatutil.

    Parameters
    ----------
    ra : float
        Right Ascension (degrees)
    dec : float
        Declination (degrees)

    Returns
    -------
    numpy array ECI (3-vector)
    """
    r = np.radians(ra)
    d = np.radians(dec)
    return np.array([np.cos(r) * np.cos(d), np.sin(r) * np.cos(d), np.sin(d)])


def nominal_roll(ra, dec, time=None, sun_ra=None, sun_dec=None, method=None):
    """
    Calculate the nominal roll angle for the given spacecraft attitude.

    If both ``sun_ra`` and ``sun_dec`` are provided then those are used for the Sun
    position; otherwise ``time`` is used to compute the Sun position.

    Parameters
    ----------
    ra : float
        Right ascension.
    dec : float
        Declination.
    time : CxoTimeLike, optional
        Time in any Chandra.Time format.
    sun_ra : float, optional
        Sun right ascension (instead of using `time`).
    sun_dec : float, optional
        Sun declination (instead of using `time`).
    method : str, optional.
        Method for calculating sun position. Valid options are "accurate", "fast".

    Returns
    -------
    float
        Nominal roll angle in the range of 0-360 degrees.

    Examples
    --------
    >>> nominal_roll(205.3105, -14.6925, time='2011:019:20:51:13')
    68.830209134280665    # vs. 68.80 for obsid 12393 in JAN1711A
    """
    if sun_ra is None or sun_dec is None:
        sun_ra, sun_dec = position(time, method=method)
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
    # shouldn't be needed but do it anyway
    body_z = body_z / np.sqrt(np.sum(body_z**2))
    roll = np.degrees(np.arctan2(body_y[2], body_z[2]))

    # Convert to 0-360 range (arctan2 is -pi to pi)
    roll = roll % 360

    return roll


def off_nominal_roll(att, time=None, sun_ra=None, sun_dec=None, method=None):
    """
    Calculate off-nominal roll angle for spacecraft attitude ``att``.

    If both ``sun_ra`` and ``sun_dec`` are provided then those are used for the Sun
    position; otherwise ``time`` is used to compute the Sun position.

    ``method`` sets the method for computing the sun position. See ``position()`` for
    details.

    Off-nominal roll is defined as ``roll - nominal roll``.

    Examples
    --------
    >>> att = (198.392135, 36.594359, 33.983322)  # RA, Dec, Roll of obsid 16354
    >>> ska_sun.off_nominal_roll(att, '2015:335:00:00:00')
    -12.22401097980562

    Parameters
    ----------
    att : QuatLike
        Chandra attitude.
    time : CxoTimeLike (optional)
        Time of observation.
    sun_ra: float, optional
        Sun RA (deg) instead of time.
    sun_dec : float, optional
        Sun Dec (deg) instead of time.
    method : str, optional
        Method for calculating sun position. Valid options are "accurate", "fast".

    Returns
    -------
    float
        Off-nominal roll angle in the range of -180 to 180 degrees.
    """
    if sun_ra is None or sun_dec is None:
        sun_ra, sun_dec = position(time, method=method)

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


def get_sun_pitch_yaw(
    ra, dec, time=None, sun_ra=None, sun_dec=None, coord_system="spacecraft"
):
    """Get Sun pitch and yaw angles of Sky coordinate(s).

    If both ``sun_ra`` and ``sun_dec`` are provided then those are used for the Sun
    position; otherwise ``time`` is used to compute the Sun position.

    The yaw coordinate system can be "spacecraft" (default) or "ORviewer". The
    "spacecraft" system matches the engineering definition of yaw about the Sun at
    a sun pitch of 90 degrees: zero yaw is defined as the ECI z-axis if the Sun is at
    exactly RA = Dec = 0.

    The example below uses real telemetry from the 2024:036 NSM recovery activity. At around
    2024:036:01:32:00.000, a "positive yaw bias of 0.025 deg/s for 30 minutes" was
    applied. The yaw maneuver converged to a stable attitude around  0210z. The
    expectation is that pitch is 90.0 and yaw *increases* by 45 degrees of the maneuver.
    ::

      >>> from ska_sun import get_sun_pitch_yaw
      >>> from Quaternion import Quat

      # From telemetry at 2024:036 0130z and 0210z
      >>> att0 = Quat([60.35823393, -34.92322161, 291.13817936])
      >>> att1 = Quat([111.94789334, -71.01495527, 335.27170577])

      >>> get_sun_pitch_yaw(att0.ra, att0.dec, time="2024:036:01:30:00")
      (90.60565371045911, 126.82092681074306)
      >>> get_sun_pitch_yaw(att1.ra, att1.dec, time="2024:036:02:10:00")
      (90.97070080025568, 171.81963428481384)

    Parameters
    ----------
    ra : float, ndarray
        RA(s), degrees
    dec : float, ndarray
        Dec(s), degrees
    time : date-like, optional
        Date of observation.
    sun_ra : float, optional
        RA of sun.  If not given, use estimated sun RA at ``date``.
    sun_dec : float, optional
        Dec of sun.  If not given, use estimated sun dec at ``date``.
    coord_system : str, optional
        Coordinate system for yaw ("spacecraft" | "ORviewer", default="spacecraft").

    Returns
    -------
    pitch : float, ndarray
        Sun pitch angle(s) in degrees.
    yaw : float, ndarray
        Sun yaw angle(s) in degrees.
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
    sign = 1 if coord_system == "spacecraft" else -1
    yaw = sign * np.arctan2(att_sun[..., 1], att_sun[..., 2])  # -pi <= yaw < pi
    yaw = yaw % (2 * np.pi)  # 0 <= yaw < 2pi

    return np.rad2deg(pitch), np.rad2deg(yaw)


def apply_sun_pitch_yaw(
    att, pitch=0, yaw=0, time=None, sun_ra=None, sun_dec=None, coord_system="spacecraft"
):
    """Apply pitch(es) and yaw(s) about Sun line to an attitude.

    If both ``sun_ra`` and ``sun_dec`` are provided then those are used for the Sun
    position; otherwise ``time`` is used to compute the Sun position.

    The yaw coordinate system can be "spacecraft" (default) or "ORviewer". The
    "spacecraft" system matches the engineering definition of yaw about the Sun at
    a sun pitch of 90 degrees: zero yaw is defined as the ECI z-axis if the Sun is at
    exactly RA = Dec = 0.

    The example below uses real telemetry from the 2024:036 NSM recovery activity. At
    around 2024:036:01:32:00.000, a "positive yaw bias of 0.025 deg/s for 30 minutes"
    was applied. The yaw maneuver converged to a stable attitude around  0210z. The
    expectation that the computed attitude matches the 0210z attitude from telemetry
    to within about a degree.
    ::

      >>> from ska_sun import apply_sun_pitch_yaw

      >>> # From telemetry at 2024:036 0130z and 0210z
      >>> att0 = Quat([60.35823393, -34.92322161, 291.13817936])
      >>> att1 = Quat([111.94789334, -71.01495527, 335.27170577])

      # Apply yaw bias to the 0130z attitude using sun position midway.
      >>> att1_apply = apply_sun_pitch_yaw(att0, pitch=0, yaw=45, time="2024:036:01:47:00")
      >>> att1_apply.equatorial
      array([111.44329433, -71.34681456, 335.48326522])

    Parameters
    ----------
    att : Quaternion-like
        Attitude(s) to be rotated.
    pitch : float, ndarray
        Sun pitch offsets (deg)
    yaw : float, ndarray
        Sun yaw offsets (deg)
    time : CxoTimeLike, optional
        Time for sun position.
    sun_ra : float, optional
        RA of sun.  If not given, use estimated sun RA at ``time``.
    sun_dec : float, optional
        Dec of sun.  If not given, use estimated sun dec at ``time``.
    coord_system : str, optional
        Coordinate system for yaw ("spacecraft" | "ORviewer", default="spacecraft").

    Returns
    -------
    Quat
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
    if coord_system == "spacecraft":
        yaws = -yaws

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


def get_att_for_sun_pitch_yaw(
    pitch: float,
    yaw: float,
    time: CxoTimeLike = None,
    sun_ra: float | None = None,
    sun_dec: float | None = None,
    coord_system: str = "spacecraft",
    off_nom_roll: float = 0,
) -> Quat:
    """Get sun-pointed attitude for given sun pitch and yaw angles.

    This function is the inverse of ``get_sun_pitch_yaw()``, where the attitude is
    constrained to have the specified ``off_nom_roll`` angle (default=0).

    Parameters
    ----------
    pitch : float
        Sun pitch angle (deg)
    yaw : float
        Sun yaw angle (deg)
    time : CxoTimeLike, optional
        Time for sun position.
    sun_ra : float, optional
        Sun RA (deg) instead of time.
    sun_dec : float, optional
        Sun Dec (deg) instead of time.
    coord_system : str, optional
        Coordinate system for yaw ("spacecraft" | "ORviewer", default="spacecraft").
    off_nom_roll : float, optional
        Off-nominal roll angle (deg)

    Returns
    -------
    att : Quat
        Attitude quaternion.
    """
    if sun_ra is None or sun_dec is None:
        sun_ra, sun_dec = position(time)

    # Generate an attitude pointed at the north ecliptic pole. Since the sun is near the
    # ecliptic plane this never has numerical problems and pitch0 is around 90 degress.
    # Then apply the appropriate pitch and yaw offsets to get to the desired sun pitch
    # and yaw.
    roll = nominal_roll(0, 90, sun_ra=sun_ra, sun_dec=sun_dec) + off_nom_roll
    att0 = Quat([0, 90, roll])
    pitch0, yaw0 = get_sun_pitch_yaw(
        att0.ra, att0.dec, sun_ra=sun_ra, sun_dec=sun_dec, coord_system=coord_system
    )
    att = apply_sun_pitch_yaw(
        att0,
        pitch=pitch - pitch0,
        yaw=yaw - yaw0,
        sun_ra=sun_ra,
        sun_dec=sun_dec,
        coord_system=coord_system,
    )
    return att


def get_nsm_attitude(att: QuatLike, time: CxoTimeLike, pitch: float = 90) -> Quat:
    """
    Calculate the closest Normal Sun Mode attitude from starting attitude.

    The default is for a NSM pitch of 90 degrees (pure minus-Z at sun). An arbitrary
    offset pitch angle can be specified.

    The calculation is based on the Chandra - sun vector in ECI. The function defines
    the vector in the body frame that will be pointed at the sun. The normal sun mode
    maneuver is then calculated as the shortest maneuver that points that vector at the
    Sun. Note that for off-nominal roll, this is NOT a pure pitch maneuver.

    Parameters
    ----------
    att : Quat
        Attitude that can initialize a Quat().
    time : CxoTimeLike
        Time in a valid Chandra.Time format.
    pitch : float, optional
        NSM pitch angle in degrees. The default is 90.

    Returns
    -------
    Quat
        NSM attitude quaternion.
    """
    # Calc Chandra - sun vector in ECI (ignore CXO orbit)
    (sun_ra, sun_dec) = position(time)
    sun_eci = np.array(_radec2eci(sun_ra, sun_dec))

    cxo_att = Quat(att)

    # Define the vector in body frame that will be pointed at the sun.
    # Pitch=90 (pure minus-Z at sun) : [0, 0, -1]
    # Pitch=160 (toward aft of spacecraft) : [-0.940, 0, -0.342]
    pitch_rad = np.deg2rad(pitch)
    vec_sun_body = np.array([np.cos(pitch_rad), 0, -np.sin(pitch_rad)])
    cxo_z_eci = np.dot(cxo_att.transform, vec_sun_body)

    # Calculate the normal sun mode maneuver as the shortest manvr that points
    # the Chandra -Z axis at the Sun. Note that for off-nominal roll this is NOT a pure
    # pitch maneuver.
    rot_angle = np.arccos(np.dot(cxo_z_eci, sun_eci))
    rot_axis = np.cross(cxo_z_eci, sun_eci)
    norm2 = np.dot(rot_axis, rot_axis)
    if norm2 < 1e-16:
        rot_axis = np.array([1.0, 0.0, 0.0])
    else:
        rot_axis /= np.sqrt(norm2)

    sra = np.sin(rot_angle / 2) * rot_axis
    manvr = Quat([sra[0], sra[1], sra[2], np.cos(rot_angle / 2)])

    return manvr * cxo_att
