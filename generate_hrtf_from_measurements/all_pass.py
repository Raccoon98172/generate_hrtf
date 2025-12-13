import numpy as np
from scipy.signal import lfilter, find_peaks
from group_delay import compute_excess_group_delay

"""
References:
  [1] Reddy, S. C., & Hegde, R. M. (2015). Minimum-Phase HRTF Modelling of Pinna Spectral Notches
      using Group Delay Decomposition. IET Research Journals.
  [2] Toledo, D., & Møller, H. (2008). Audibility of high Q-factor all-pass components in HRTFs.
      AES Convention Paper 7565.
  [3] Plogsties, J., et al. (2000). Audibility of all-pass components in head-related transfer functions.
      AES Convention Paper 5132.
  [4] Nam, J., et al. (2008). On the Minimum-Phase Nature of Head-Related Transfer Functions.
      AES Convention Paper 7565.
"""


def generate_second_order_allpass(f0, r, fs):
    """
    Returns coefficients (b, a) for a Second-Order All-Pass Filter.

    Transfer function:
    H(z) = (r^2 - 2r*cos(w0)*z^-1 + z^-2) / (1 - 2r*cos(w0)*z^-1 + r^2*z^-2)

    Args:
        f0: Center frequency (Hz)
        r: Pole radius (0-1), related to Q-factor
        fs: Sampling rate (Hz)

    Returns:
        b, a: Numerator and denominator coefficients
    """
    if r <= 0.01 or f0 <= 100:
        return np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])

    w0 = 2 * np.pi * f0 / fs
    cos_w0 = np.cos(w0)

    r_stable = np.clip(r, 0.0, 0.99)

    # numerator: [r^2, -2r*cos(w0), 1]
    b0 = r_stable**2
    b1 = -2 * r_stable * cos_w0
    b2 = 1.0

    # denominator: [1, -2r*cos(w0), r^2]
    a0 = 1.0
    a1 = -2 * r_stable * cos_w0
    a2 = r_stable**2

    b = np.array([b0, b1, b2])
    a = np.array([a0, a1, a2])

    return b, a


def apply_all_pass_filter(signal, f0, r, fs):
    """
    Applies the all-pass filter to a 1D signal (HRIR).

    Args:
        signal: Input impulse response
        f0: Center frequency (Hz)
        r: Pole radius
        fs: Sampling rate (Hz)

    Returns:
        Filtered signal with all-pass applied
    """

    if r <= 0.01:
        return signal

    b, a = generate_second_order_allpass(f0, r, fs)
    return lfilter(b, a, signal)


def analyze_multiple_allpass_sections(
    freqs, measured_spl, measured_phase_deg, max_sections=3, min_separation_hz=2000
):
    """
    Apply multiple all-pass sections in cascade to a signal.

    Args:
        signal: Input HRIR
        sections: List of parameters
        fs: Sampling rate (Hz)

    Returns:
        Filtered signal (Cascaded biquads)
    """

    gd_freqs, gd_smooth_ms = compute_excess_group_delay(
        freqs, measured_spl, measured_phase_deg
    )

    search_mask = (gd_freqs >= 3000) & (gd_freqs <= 18000)

    if np.sum(search_mask) < 10:
        return [], gd_freqs, gd_smooth_ms

    gd_roi = gd_smooth_ms[search_mask]
    freq_roi = gd_freqs[search_mask]

    gd_median = np.median(gd_roi)
    gd_std = np.std(gd_roi)

    prominence_threshold = max(gd_std * 0.2, 0.1)  # 0.1 ms minimum prominence

    freq_resolution = freq_roi[1] - freq_roi[0] if len(freq_roi) > 1 else 100
    min_distance_samples = int(min_separation_hz / freq_resolution)
    min_distance_samples = max(min_distance_samples, 1)

    peaks, properties = find_peaks(
        gd_roi,
        prominence=prominence_threshold,
        width=2,
        distance=min_distance_samples,
    )

    if len(peaks) == 0:
        return [], gd_freqs, gd_smooth_ms

    sections = []

    notch_search_radius_hz = 200
    notch_depth_threshold_ms = -0.4

    search_radius_samples = int(notch_search_radius_hz / freq_resolution)

    for i, peak_idx in enumerate(peaks):

        start_idx = max(0, peak_idx - search_radius_samples)
        end_idx = min(len(gd_roi), peak_idx + search_radius_samples + 1)
        neighborhood_gd = gd_roi[start_idx:end_idx]

        min_gd_in_neighborhood = np.min(neighborhood_gd)

        if min_gd_in_neighborhood < notch_depth_threshold_ms:
            # reject the peak if a deep negative Group Delay dip is found nearby
            # this indicates a spectral notch artifact, not a true all-pass characteristic
            continue

        f0 = freq_roi[peak_idx]
        peak_gd_ms = gd_roi[peak_idx]
        prominence = properties["prominences"][i]

        # ignore peaks smaller than 0.25 ms (inaudible)
        if peak_gd_ms < 0.25:
            continue

        fs = 48000.0
        peak_gd_sec = peak_gd_ms / 1000.0
        T_s = 1.0 / fs

        A = peak_gd_sec / T_s

        if A <= 1.0:
            r = 0.0
        else:
            r = (A - 1.0) / (A + 1.0)
            r = np.clip(r, 0.0, 0.99)

        if r > 0.05:
            sections.append((f0, r, prominence))

    sections.sort(key=lambda x: x[2], reverse=True)

    sections = sections[:max_sections]

    return sections, gd_freqs, gd_smooth_ms


def apply_multiple_allpass_filters(signal, sections, fs):
    """
    Apply multiple all-pass sections in cascade to a signal.

    Args:
        signal: Input HRIR
        sections: List of (f0, r) or (f0, r, prominence) tuples
        fs: Sampling rate (Hz)

    Returns:
        Filtered signal with all all-pass sections applied in cascade
    """

    if len(sections) == 0:
        return signal

    if len(sections[0]) == 3:
        allpass_pairs = [(f0, r) for f0, r, _ in sections]
    else:
        allpass_pairs = sections

    allpass_pairs.sort(key=lambda x: x[0])

    output = signal.copy()

    for f0, r in allpass_pairs:
        if r > 0.01:
            output = apply_all_pass_filter(output, f0, r, fs)

    return output


def check_multiple_allpass_audibility(sections_left, sections_right, fs=48000):
    """
    Check if the detected all-pass sections are likely to be audible.

    Returns:
        bool: True if audible/significant, False otherwise.
    """

    def extract_f0_r(sections):
        if not sections:
            return []
        if len(sections[0]) == 3:
            return [(x[0], x[1]) for x in sections]
        return sections

    pairs_left = extract_f0_r(sections_left)
    pairs_right = extract_f0_r(sections_right)

    if not pairs_left and not pairs_right:
        return False

    def total_gd_at_dc(pairs, fs):
        total = 0.0
        for f0, r in pairs:
            if r > 0.01:
                w0 = 2 * np.pi * f0 / fs
                denom = 1 + r**2 - 2 * r * np.cos(w0)
                if denom != 0:
                    gd = 2 * (1 - r**2) / denom
                    total += gd
        return total

    gd_left = total_gd_at_dc(pairs_left, fs)
    gd_right = total_gd_at_dc(pairs_right, fs)

    IGD_0_us = (abs(gd_left - gd_right) / fs) * 1e6
    if IGD_0_us > 30:
        return True

    all_pairs = pairs_left + pairs_right
    max_r = max(r for f0, r in all_pairs)

    if max_r > 0.5:
        return True

    if len(all_pairs) >= 4 and max_r > 0.2:
        return True

    return False


def report_allpass_sections(
    az,
    el,
    sections_l,
    sections_r,
    debug_freqs_l,
    debug_gd_l,
    debug_freqs_r,
    debug_gd_r,
    visualizer,
    is_debug,
):
    if not is_debug:
        return

    print(10 * "-")

    print(f"[{az} deg, {el} deg]")

    def _print_side(label, secs):
        print(f"  {label}: {len(secs)} sections")
        for i, (f0, r, prom) in enumerate(secs):
            Q = (1 + r) / (1 - r) if r > 0.01 else 0
            print(f"    {i+1}) f={f0:.0f} Hz, r={r:.3f}, Q={Q:.1f}")

    _print_side("LEFT", sections_l)
    _print_side("RIGHT", sections_r)

    print(10 * "-")

    if (az == 30 or az == 330) and el == 0:
        if len(sections_l) > 0:
            visualizer.plot_group_delay_extraction(
                debug_freqs_l, debug_gd_l, sections_l, az, el, "left"
            )
        if len(sections_r) > 0:
            visualizer.plot_group_delay_extraction(
                debug_freqs_r, debug_gd_r, sections_r, az, el, "right"
            )
