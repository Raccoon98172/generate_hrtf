import numpy as np
from scipy import signal, interpolate

POINTS_PER_OCTAVE = 192
SMOOTHING_FACTOR = 0.02


def interpolate_to_dense_grid(frequencies, magnitudes, phases):
    """
    Interpolates data to a dense logarithmic grid.
    """
    f_min = frequencies[0]
    f_max = frequencies[-1]

    n_octaves = np.log2(f_max / f_min)
    n_points = int(n_octaves * POINTS_PER_OCTAVE)

    freq_dense = np.logspace(np.log10(f_min), np.log10(f_max), n_points)

    mag_interp = interpolate.PchipInterpolator(frequencies, magnitudes)
    phase_interp = interpolate.PchipInterpolator(frequencies, phases)

    mag_dense = mag_interp(freq_dense)
    phase_dense = phase_interp(freq_dense)

    return freq_dense, mag_dense, phase_dense


def extend_spectrum_for_hilbert(frequencies, magnitudes):
    """
    Extends spectrum for correct Hilbert transform (reduces edge artifacts).
    """
    f_low = np.logspace(np.log10(10), np.log10(frequencies[0]), 100)[:-1]
    mag_low = np.ones(len(f_low)) * magnitudes[0]

    f_high = np.logspace(np.log10(frequencies[-1]), np.log10(96000), 100)[1:]
    mag_high = np.ones(len(f_high)) * magnitudes[-1]

    freq_extended = np.concatenate([f_low, frequencies, f_high])
    mag_extended = np.concatenate([mag_low, magnitudes, mag_high])

    return freq_extended, mag_extended


def calculate_minimum_phase_hilbert(frequencies, magnitudes):
    """
    Calculates the minimum phase response using the Hilbert transform method.

    Relation:
    Phase_min(w) = -Imag(Hilbert(log(|H(w)|)))

    Args:
        frequencies: Frequency array (Hz)
        magnitudes: Magnitude response (dB)

    Returns:
        min_phase_deg: Minimum phase response in degrees
    """
    freq_ext, mag_ext = extend_spectrum_for_hilbert(frequencies, magnitudes)

    mag_linear = 10 ** (mag_ext / 20.0)
    log_mag = np.log(np.maximum(mag_linear, 1e-10))

    analytic_signal = signal.hilbert(log_mag)
    min_phase_rad = -np.imag(analytic_signal)

    idx_start = np.searchsorted(freq_ext, frequencies[0])
    idx_end = np.searchsorted(freq_ext, frequencies[-1])

    min_phase_rad = min_phase_rad[idx_start : idx_end + 1]

    if len(min_phase_rad) != len(frequencies):
        freq_range = freq_ext[idx_start : idx_end + 1]
        min_phase_rad = np.interp(frequencies, freq_range, min_phase_rad)

    return np.rad2deg(min_phase_rad)


def unwrap_phase(phase_degrees):
    phase_rad = np.deg2rad(phase_degrees)
    unwrapped_rad = np.unwrap(phase_rad)
    return np.rad2deg(unwrapped_rad)


def remove_linear_phase(frequencies, phase_unwrapped):
    """
    Removes the linear phase component (pure delay) from the phase response.

    Args:
        frequencies: Frequency array (Hz)
        phase_unwrapped: Unwrapped phase (degrees)

    Returns:
        Phase in degrees with linear slope removed
    """
    coeffs = np.polyfit(frequencies, phase_unwrapped, 1)
    linear_phase = np.polyval(coeffs, frequencies)
    return phase_unwrapped - linear_phase


def calculate_group_delay(frequencies, phase_degrees):
    """
    Calculates Group Delay from the phase response.

    Formula:
    GD(w) = -d(phi)/dw

    Args:
        frequencies: Frequency array (Hz)
        phase_degrees: Phase response (degrees)

    Returns:
        group_delay_ms: Group delay in MILLISECONDS
    """
    phase_rad = np.deg2rad(phase_degrees)
    dphase = np.gradient(phase_rad, frequencies)
    group_delay_ms = -dphase / (2 * np.pi) * 1000
    return group_delay_ms


def smooth_group_delay(frequencies, group_delay):
    """
    Applies frequency-dependent smoothing to the group delay.

    Args:
        frequencies: Frequency array (Hz)
        group_delay: Raw group delay values

    Returns:
        gd_smooth: Smoothed group delay
    """
    gd_smooth = group_delay.copy()
    log_freq = np.log10(frequencies)

    for i in range(2, len(group_delay) - 2):
        window = int(len(frequencies) * SMOOTHING_FACTOR / (1 + log_freq[i]))
        window = max(3, min(window, 11))

        if window % 2 == 0:
            window += 1

        half_window = window // 2
        start_idx = max(0, i - half_window)
        end_idx = min(len(group_delay), i + half_window + 1)

        gd_smooth[i] = np.median(group_delay[start_idx:end_idx])

    return gd_smooth


def compute_excess_group_delay(freqs, spl, phase):
    """
    Computes the smoothed Excess Group Delay.

    The process isolates the phase contribution of All-Pass components by:
    1. Calculating Minimum Phase from Magnitude.
    2. Subtracting Minimum Phase and Linear Phase (delay) from Total Phase.
    3. Computing the Group Delay of the remaining (Excess) Phase.

    Args:
        freqs: Frequency array (Hz)
        spl: Magnitude (dB)
        phase: Phase (degrees)

    Returns:
        freq_dense: High resolution frequency array
        gd_excess_smooth: Smoothed excess group delay in MILLISECONDS
    """
    phase_raw_unwrapped = unwrap_phase(phase)

    freq_dense, mag_dense, phase_dense = interpolate_to_dense_grid(
        freqs, spl, phase_raw_unwrapped
    )

    phase_unwrapped = phase_dense

    phase_minimum = calculate_minimum_phase_hilbert(freq_dense, mag_dense)

    phase_unwrapped_nonlinear = remove_linear_phase(freq_dense, phase_unwrapped)

    excess_phase = phase_unwrapped_nonlinear - phase_minimum

    gd_excess = calculate_group_delay(freq_dense, excess_phase)

    gd_excess_smooth = smooth_group_delay(freq_dense, gd_excess)

    return freq_dense, gd_excess_smooth
