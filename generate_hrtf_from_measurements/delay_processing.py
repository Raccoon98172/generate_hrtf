import numpy as np


def calculate_woodworth_itd_array(positions, head_width_cm, strength=1.0):
    """
    Calculates Interaural Time Difference (ITD) values based on the Woodworth spherical head model.

    Formula:
    ITD = -(r / c) * (alpha + sin(alpha))

    where:
    r = head_radius
    c = speed of sound
    alpha = lateral angle derived from Cartesian Y-coordinate

    Args:
        positions: List or array of [azimuth, elevation] pairs in degrees.
        head_width_cm: Diameter of the head in centimeters.
        strength: Scaling factor for the resulting ITD (default 1.0).

    Returns:
        np.array: Array of ITD values in seconds corresponding to each position.
    """
    c = 34300.0
    r_head = head_width_cm / 2.0
    itds = []
    for i in range(len(positions)):
        az = positions[i][0]
        el = positions[i][1]
        _, y, _ = sph2cart(az, el, 1.0)
        y = np.clip(y, -1.0, 1.0)
        alpha = np.arcsin(y)
        itd_val = -1.0 * (r_head / c) * (alpha + np.sin(alpha))
        itds.append(itd_val * strength)
    return np.array(itds)


def apply_integer_delay(signal, delay_samples):
    if delay_samples == 0:
        return signal

    new_signal = np.zeros_like(signal)
    if delay_samples > 0 and delay_samples < len(signal):
        new_signal[delay_samples:] = signal[:-delay_samples]

    return new_signal


def sph2cart(az, el, r=1.0):
    """
    Converts spherical coordinates to Cartesian coordinates.

    Args:
        az: Azimuth angle in degrees.
        el: Elevation angle in degrees.
        r: Radius (distance from origin), default is 1.0.

    Returns:
        x, y, z: Cartesian coordinates.
    """
    az_rad = np.deg2rad(az)
    el_rad = np.deg2rad(el)
    x = r * np.cos(el_rad) * np.cos(az_rad)
    y = r * np.cos(el_rad) * np.sin(az_rad)
    z = r * np.sin(el_rad)
    return x, y, z
