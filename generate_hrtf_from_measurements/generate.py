# generate.py

import numpy as np
import sofar as sf
import os
import re
from scipy.signal import medfilt
from scipy.spatial import cKDTree
from scipy.interpolate import Rbf

from log_smoothing import apply_trend_balance
from hrtf_plots import HRTFVisualizer
from hrtf_lfe import extend_low_frequencies

visualizer = HRTFVisualizer(output_folder="hrtf_processing_plots")

# Settings

N_SAMPLES = 2048
TARGET_FS = 48000
FREQ_START = 550
FREQ_END = 19500

LFE_CROSSOVER_FREQ = 550.0
LFE_TRANSITION_HZ = 450.0

HEAD_WIDTH_CM = 13.0
HEAD_DEPTH_CM = 20.5

ITD_STRENGTH = 1.0

APPLY_DIFFUSE_EQ = False
DESPIKE_SOURCE = False
FORCE_CENTER_SYMMETRY = False

APPLY_TREND_BALANCE = False

ENABLE_INTERPOLATION = False
BASE_GRID_STEP = 5.0
ELEVATION_STEP = 10.0

MIRROR_MODE = "NONE"  # "NONE" / "LEFT_TO_RIGHT" / "RIGHT_TO_LEFT"

INPUT_FOLDER = "my_hrtf"
OUTPUT_SOFA = "my_hrtf.sofa"


def parse_rew_file(filepath, freq_start, freq_end):
    freqs, spl = [], []
    if not os.path.exists(filepath):
        return np.array([]), np.array([])
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("*"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    f_val = float(parts[0])
                    s_val = float(parts[1])
                    if freq_start <= f_val <= freq_end:
                        freqs.append(f_val)
                        spl.append(s_val)
                except ValueError:
                    continue
    return np.array(freqs), np.array(spl)


def parse_filename(filename):
    match = re.match(r"(\d+)_(left|right)(?:_([+-]?\d+)_elevation)?\.txt", filename)
    if match:
        return (
            int(match.group(1)),
            int(match.group(3)) if match.group(3) else 0,
            match.group(2),
        )
    return None, None, None


def apply_mirroring_logic(data_map, mode):
    if mode == "NONE":
        return data_map
    print(f"Mirroring: {mode}")
    new_map = data_map.copy()
    existing_coords = set(new_map.keys())
    for (az, el), data in data_map.items():
        if az == 0 or az == 180:
            continue
        should_mirror = False
        if mode == "LEFT_TO_RIGHT" and (0 < az < 180):
            should_mirror = True
        elif mode == "RIGHT_TO_LEFT" and (180 < az < 360):
            should_mirror = True
        if should_mirror:
            mirror_az = (360 - az) % 360
            if (mirror_az, el) not in existing_coords:
                if "left" in data and "right" in data:
                    new_map[(mirror_az, el)] = {
                        "left": data["right"],
                        "right": data["left"],
                    }
    return new_map


def apply_smooth_symmetry_blend(data_map, blend_start=400, blend_end=1200):
    """
    Smooth channel blending.
    - Below blend_start: Perfect symmetry (Left = Right = Average).
    - From blend_start to blend_end: Smooth crossfade from Average to Raw.
    - Above blend_end: Original data.
    This removes any bends and sharp edges.
    """
    print(f"Applying Smooth Symmetry Blend ({blend_start}-{blend_end} Hz)...")

    processed_pairs = set()

    def blend_arrays(freqs, spl_1, spl_2):
        avg = (spl_1 + spl_2) / 2.0

        weights = (freqs - blend_start) / (blend_end - blend_start)
        weights = np.clip(weights, 0.0, 1.0)

        new_spl_1 = avg * (1.0 - weights) + spl_1 * weights
        new_spl_2 = avg * (1.0 - weights) + spl_2 * weights

        return new_spl_1, new_spl_2

    for (az, el), data in data_map.items():
        if az == 0 or az == 180:
            if "left" in data and "right" in data:
                f_l, s_l = data["left"]
                f_r, s_r = data["right"]

                s_l, s_r = blend_arrays(f_l, s_l, s_r)
                data["left"] = (f_l, s_l)
                data["right"] = (f_r, s_r)

    for az, el in list(data_map.keys()):
        if az == 0 or az == 180:
            continue
        if (az, el) in processed_pairs:
            continue

        mirror_az = (360 - az) % 360
        if (mirror_az, el) in data_map:
            d_main = data_map[(az, el)]
            d_mirror = data_map[(mirror_az, el)]

            if (
                "left" in d_main
                and "right" in d_main
                and "left" in d_mirror
                and "right" in d_mirror
            ):

                f_m, s_mL = d_main["left"]
                f_mr, s_mR = d_mirror["right"]
                s_mL, s_mR = blend_arrays(f_m, s_mL, s_mR)
                d_main["left"] = (f_m, s_mL)
                d_mirror["right"] = (f_mr, s_mR)

                f_m, s_mR = d_main["right"]
                f_mr, s_mL = d_mirror["left"]
                s_mR, s_mL = blend_arrays(f_m, s_mR, s_mL)
                d_main["right"] = (f_m, s_mR)
                d_mirror["left"] = (f_mr, s_mL)

                processed_pairs.add((az, el))
                processed_pairs.add((mirror_az, el))

    return data_map


def sph2cart(az, el, r=1.0):
    az_rad = np.deg2rad(az)
    el_rad = np.deg2rad(el)
    x = r * np.cos(el_rad) * np.cos(az_rad)
    y = r * np.cos(el_rad) * np.sin(az_rad)
    z = r * np.sin(el_rad)
    return x, y, z


def calculate_woodworth_itd_array(positions, head_width_cm, strength=1.0):
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


def generate_grid(base_step=5.0, el_step=10.0):
    grid_pos = []
    el_range = np.arange(-40, 91, el_step)
    for el in el_range:
        if np.isclose(abs(el), 90):
            grid_pos.append([0, el, 1.0])
            continue
        scale_factor = np.cos(np.deg2rad(el))
        current_step = base_step / max(0.01, scale_factor)
        az_range = np.arange(0, 360, current_step)
        for az in az_range:
            grid_pos.append([az, el, 1.0])
    return np.array(grid_pos)


def interpolate_thin_plate(source_pos, source_data, target_pos):
    print(f"(Thin-Plate) for {source_data.shape[1]} features...")
    sx, sy, sz = sph2cart(source_pos[:, 0], source_pos[:, 1])
    tx, ty, tz = sph2cart(target_pos[:, 0], target_pos[:, 1])
    rbf = Rbf(sx, sy, sz, source_data, function="thin_plate", smooth=0, mode="N-D")
    return rbf(tx, ty, tz)


def convert_to_min_phase_pure(magnitudes, fs, n_samples):
    n_fft = (magnitudes.shape[1] - 1) * 2
    cep_window = np.zeros(n_fft)
    cep_window[0] = 1.0
    cep_window[1 : n_fft // 2] = 2.0
    cep_window[n_fft // 2] = 1.0

    final_spectra = np.zeros_like(magnitudes, dtype=np.complex128)

    for i in range(magnitudes.shape[0]):
        for e in range(2):
            mag = magnitudes[i, :, e]
            mag = np.maximum(mag, 1e-9)

            log_mag = np.log(mag)
            cep = np.fft.irfft(log_mag, n=n_fft)
            cep = cep * cep_window
            min_ph_spec = np.exp(np.fft.rfft(cep))
            final_spectra[i, :, e] = min_ph_spec

    return final_spectra


def apply_integer_delay(signal, delay_samples):
    if delay_samples == 0:
        return signal

    new_signal = np.zeros_like(signal)
    if delay_samples > 0 and delay_samples < len(signal):
        new_signal[delay_samples:] = signal[:-delay_samples]

    return new_signal


def run_processing():
    print(f"Sampling: {N_SAMPLES} samples @ {TARGET_FS} Hz")

    if not os.path.exists(INPUT_FOLDER):
        print(f"Input folder not found: {INPUT_FOLDER}")
        return

    # Load
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")]
    data_map = {}
    for f in files:
        az, el, ch = parse_filename(f)
        if az is None:
            continue
        freqs, spl = parse_rew_file(os.path.join(INPUT_FOLDER, f), FREQ_START, FREQ_END)
        if len(freqs) > 0:
            if (az, el) not in data_map:
                data_map[(az, el)] = {}
            data_map[(az, el)][ch] = (freqs, spl)

    visualizer.plot_loaded_data(data_map)

    data_map = apply_mirroring_logic(data_map, MIRROR_MODE)

    data_map = apply_smooth_symmetry_blend(data_map, blend_start=400, blend_end=1200)

    # TREND (Impulcifer)
    if APPLY_TREND_BALANCE:
        visualizer.plot_trend_correction(data_map, TARGET_FS, N_SAMPLES)
        data_map = apply_trend_balance(data_map, TARGET_FS, N_SAMPLES)
        visualizer.plot_after_trend(data_map)

    # Source Data Generation
    src_pos, src_mags_l, src_mags_r = [], [], []
    target_freqs = np.fft.rfftfreq(N_SAMPLES, 1.0 / TARGET_FS)

    for az, el in sorted(data_map.keys()):
        if "left" in data_map[(az, el)] and "right" in data_map[(az, el)]:
            fl, spl_l = data_map[(az, el)]["left"]
            fr, spl_r = data_map[(az, el)]["right"]

            full_spl_l, full_spl_r = extend_low_frequencies(
                left_measured_freqs=fl,
                left_measured_spl=spl_l,
                right_measured_freqs=fr,
                right_measured_spl=spl_r,
                target_freqs=target_freqs,
                crossover_freq=LFE_CROSSOVER_FREQ,
                transition_width_hz=LFE_TRANSITION_HZ,
            )

            linear_mag_l = 10 ** (full_spl_l / 20.0)
            linear_mag_r = 10 ** (full_spl_r / 20.0)

            if DESPIKE_SOURCE:
                linear_mag_l = medfilt(linear_mag_l, kernel_size=3)
                linear_mag_r = medfilt(linear_mag_r, kernel_size=3)

            src_pos.append([az, el, 1.0])
            src_mags_l.append(linear_mag_l)
            src_mags_r.append(linear_mag_r)

    src_pos = np.array(src_pos)
    src_mags_l = np.array(src_mags_l)
    src_mags_r = np.array(src_mags_r)

    visualizer.plot_source_magnitudes(
        src_pos, src_mags_l, src_mags_r, TARGET_FS, N_SAMPLES
    )

    visualizer.plot_symmetry_check(
        src_pos, src_mags_l, src_mags_r, TARGET_FS, N_SAMPLES
    )

    # Diffuse EQ
    if APPLY_DIFFUSE_EQ:
        print("Applying Diffuse EQ...")
        pow_l, pow_r = np.mean(src_mags_l**2, axis=0), np.mean(src_mags_r**2, axis=0)
        avg_mag = np.sqrt((pow_l + pow_r) / 2)
        avg_mag = np.maximum(avg_mag, 1e-6)

        ref_bin = int(1000 * N_SAMPLES / TARGET_FS)
        ref_bin = min(ref_bin, len(avg_mag) - 1)

        if avg_mag[ref_bin] > 0:
            avg_mag /= avg_mag[ref_bin]
            src_mags_l /= avg_mag
            src_mags_r /= avg_mag

    # Interpolation
    if ENABLE_INTERPOLATION:
        target_pos = generate_grid(BASE_GRID_STEP, ELEVATION_STEP)

        print("Interpolating Frequency Responses...")
        interp_mags_l = interpolate_thin_plate(src_pos, src_mags_l, target_pos)
        interp_mags_r = interpolate_thin_plate(src_pos, src_mags_r, target_pos)

        visualizer.plot_interpolation_comparison(
            src_pos, target_pos, src_mags_l, interp_mags_l, TARGET_FS, N_SAMPLES
        )

        sx, sy, sz = sph2cart(src_pos[:, 0], src_pos[:, 1])
        tx, ty, tz = sph2cart(target_pos[:, 0], target_pos[:, 1])
        src_cart = np.stack([sx, sy, sz], axis=1)
        target_cart = np.stack([tx, ty, tz], axis=1)

        tree = cKDTree(target_cart)
        distances, indices = tree.query(src_cart)
        for i, (dist, idx_in_grid) in enumerate(zip(distances, indices)):
            if dist < 0.02:
                interp_mags_l[idx_in_grid] = src_mags_l[i]
                interp_mags_r[idx_in_grid] = src_mags_r[i]

        print("Calculating Analytical ITD (Woodworth Model)...")
        final_itds = calculate_woodworth_itd_array(
            target_pos, HEAD_WIDTH_CM, ITD_STRENGTH
        )
        final_pos = target_pos
        final_mags = np.stack([interp_mags_l, interp_mags_r], axis=-1)

    else:
        print("Calculating Analytical ITD ...")
        final_itds = calculate_woodworth_itd_array(src_pos, HEAD_WIDTH_CM, ITD_STRENGTH)
        final_pos = src_pos
        final_mags = np.stack([src_mags_l, src_mags_r], axis=-1)

    visualizer.plot_itd_analysis(final_pos, final_itds)

    final_mags = np.maximum(final_mags, 1e-9)

    # Reconstruction
    print("Reconstructing Audio (No Delay Phase)...")
    final_spectra = convert_to_min_phase_pure(final_mags, TARGET_FS, N_SAMPLES)

    hrirs = np.fft.irfft(final_spectra, n=N_SAMPLES, axis=1)
    hrirs = np.transpose(hrirs, (0, 2, 1))

    print("Applying ITD...")
    safety_offset = 72

    for i in range(len(hrirs)):
        itd_sec = final_itds[i]
        delay_diff = int(round(abs(itd_sec) * TARGET_FS))
        delay_L = safety_offset
        delay_R = safety_offset

        if itd_sec > 0:
            delay_L += delay_diff
        elif itd_sec < 0:
            delay_R += delay_diff

        hrirs[i, 0, :] = apply_integer_delay(hrirs[i, 0, :], delay_L)
        hrirs[i, 1, :] = apply_integer_delay(hrirs[i, 1, :], delay_R)

    # Normalize
    mx = np.max(np.abs(hrirs))
    if mx > 0:
        hrirs = (hrirs / mx) * 0.95

    visualizer.plot_final_hrirs(hrirs, final_pos, TARGET_FS)

    # Save SOFA
    sofa = sf.Sofa("SimpleFreeFieldHRIR")
    sofa.Data_IR = hrirs
    sofa.Data_SamplingRate = TARGET_FS
    sofa.SourcePosition = final_pos
    sofa.ListenerPosition = [0, 0, 0]
    sofa.ReceiverPosition = [
        [0, HEAD_WIDTH_CM / 200.0, 0],
        [0, -HEAD_WIDTH_CM / 200.0, 0],
    ]
    sf.write_sofa(OUTPUT_SOFA, sofa, compression=9)
    print(f"\nDONE: {OUTPUT_SOFA}")


if __name__ == "__main__":
    run_processing()
