import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import InterpolatedUnivariateSpline


def smooth_frequency_response_log(mags_db, freqs, window_octaves=2):
    """
    Smooths the frequency response (SPL) using a logarithmic window (Trend).

    Args:
        mags_db (array): Array of SPL values (dB).
        freqs (array): Array of frequencies (Hz).
        window_octaves (float): Width of the smoothing window in octaves (Impulcifer default = 2).

    Returns:
        array: Smoothed SPL array of the same length as the input.
    """

    F_MIN, F_MAX = 20.0, 20000.0

    valid_mask = (freqs >= F_MIN) & (freqs <= F_MAX)
    f_valid = freqs[valid_mask]
    m_valid = mags_db[valid_mask]

    if len(f_valid) < 10:
        return mags_db

    NUM_POINTS = 1000
    log_freqs = np.geomspace(F_MIN, F_MAX, NUM_POINTS)

    interp_func = InterpolatedUnivariateSpline(np.log10(f_valid), m_valid, k=1)
    mags_on_log = interp_func(np.log10(log_freqs))

    total_octaves = np.log2(F_MAX / F_MIN)
    points_per_octave = NUM_POINTS / total_octaves

    window_size = int(points_per_octave * window_octaves)

    if window_size % 2 == 0:
        window_size += 1

    if window_size >= NUM_POINTS:
        window_size = NUM_POINTS - 1 if (NUM_POINTS - 1) % 2 != 0 else NUM_POINTS - 2

    mags_smoothed_log = savgol_filter(mags_on_log, window_size, polyorder=2)

    back_interp = InterpolatedUnivariateSpline(
        np.log10(log_freqs), mags_smoothed_log, k=1
    )

    safe_freqs = np.clip(freqs, F_MIN, F_MAX)
    result = back_interp(np.log10(safe_freqs))

    final_result = np.zeros_like(mags_db)

    final_result = result

    if len(freqs[freqs < F_MIN]) > 0:
        first_val = mags_smoothed_log[0]
        final_result[freqs < F_MIN] = first_val

    if len(freqs[freqs > F_MAX]) > 0:
        last_val = mags_smoothed_log[-1]
        final_result[freqs > F_MAX] = last_val

    return final_result


def apply_trend_balance(data_map, fs, n_samples):

    n_fft = n_samples

    common_freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)

    freq_mask = (common_freqs >= 20) & (common_freqs <= 20000)
    common_freqs_masked = common_freqs[freq_mask]

    azimuth_pairs = {}

    for (az, el), data in data_map.items():
        if el not in azimuth_pairs:
            azimuth_pairs[el] = {}

        if az == 0 or az == 180:
            continue

        mirror_az = (360 - az) % 360
        pair_key = tuple(sorted((az, mirror_az)))

        if pair_key not in azimuth_pairs[el]:
            azimuth_pairs[el][pair_key] = []

        azimuth_pairs[el][pair_key].append(data)

    count_corrected = 0

    for el, pairs in azimuth_pairs.items():
        for pair_key, data_points in pairs.items():
            if len(data_points) != 2:
                continue

            left_spls_on_grid = []
            right_spls_on_grid = []

            for data in data_points:
                if "left" in data and "right" in data:
                    freq_l, spl_l = data["left"]
                    freq_r, spl_r = data["right"]

                    interp_spl_l = np.interp(common_freqs_masked, freq_l, spl_l)
                    interp_spl_r = np.interp(common_freqs_masked, freq_r, spl_r)

                    left_spls_on_grid.append(interp_spl_l)
                    right_spls_on_grid.append(interp_spl_r)

            if not left_spls_on_grid or not right_spls_on_grid:
                continue

            avg_left_spl = np.mean(np.array(left_spls_on_grid), axis=0)
            avg_right_spl = np.mean(np.array(right_spls_on_grid), axis=0)

            difference_db = avg_left_spl - avg_right_spl

            correction_curve_db = smooth_frequency_response_log(
                difference_db, common_freqs_masked, window_octaves=2
            )

            az1, az2 = pair_key

            for az_to_correct in [az1, az2]:
                if (az_to_correct, el) in data_map:
                    original_freqs_r, original_spls_r = data_map[(az_to_correct, el)][
                        "right"
                    ]

                    correction_on_file_freqs = np.interp(
                        original_freqs_r, common_freqs_masked, correction_curve_db
                    )

                    corrected_spls_r = original_spls_r + correction_on_file_freqs

                    data_map[(az_to_correct, el)]["right"] = (
                        original_freqs_r,
                        corrected_spls_r,
                    )
                    count_corrected += 1

    print(f"'Trend' completed. Processed {count_corrected} files.")

    return data_map
