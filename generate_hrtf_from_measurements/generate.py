import numpy as np
import sofar as sf
import os

from log_smoothing import apply_trend_balance
from hrtf_plots import HRTFVisualizer
from hrtf_lfe import extend_low_frequencies
from all_pass import (
    analyze_multiple_allpass_sections,
    apply_multiple_allpass_filters,
    check_multiple_allpass_audibility,
    report_allpass_sections,
)
from file_parser import parse_rew_file, parse_filename
from mirroring import apply_mirroring_logic, apply_smooth_symmetry_blend
from min_phase import convert_to_min_phase
from delay_processing import calculate_woodworth_itd_array, apply_integer_delay

visualizer = HRTFVisualizer(output_folder="hrtf_processing_plots")

# Settings

HEAD_WIDTH_CM = 13.0

MIRROR_MODE = "NONE"  # "NONE" / "LEFT_TO_RIGHT" / "RIGHT_TO_LEFT"

APPLY_TREND_BALANCE = False

INPUT_FOLDER = "my_hrtf"
OUTPUT_SOFA = "my_hrtf.sofa"

# Other

DEBUG = False

MAX_ALLPASS_SECTIONS = 3
MIN_ALLPASS_SEPARATION_HZ = 2000

N_SAMPLES = 512
TARGET_FS = 48000

FREQ_START = 550
FREQ_END = 19500

LFE_CROSSOVER_FREQ = 550.0
LFE_TRANSITION_HZ = 450.0

ITD_STRENGTH = 1.0


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
        freqs, spl, phase = parse_rew_file(
            os.path.join(INPUT_FOLDER, f), FREQ_START, FREQ_END
        )
        if len(freqs) > 0:
            if (az, el) not in data_map:
                data_map[(az, el)] = {}
            data_map[(az, el)][ch] = (freqs, spl, phase)

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
    src_ap_l, src_ap_r = [], []
    target_freqs = np.fft.rfftfreq(N_SAMPLES, 1.0 / TARGET_FS)

    print("Analyzing Frequency & All-Pass Components...")

    for az, el in sorted(data_map.keys()):
        d = data_map[(az, el)]
        if "left" in d and "right" in d:
            fl, spl_l, ph_l = d["left"]
            fr, spl_r, ph_r = d["right"]

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

            src_pos.append([az, el, 1.0])
            src_mags_l.append(linear_mag_l)
            src_mags_r.append(linear_mag_r)

            # multi-section analysis
            sections_l, debug_freqs_l, debug_gd_l = analyze_multiple_allpass_sections(
                fl,
                spl_l,
                ph_l,
                max_sections=MAX_ALLPASS_SECTIONS,
                min_separation_hz=MIN_ALLPASS_SEPARATION_HZ,
            )

            sections_r, debug_freqs_r, debug_gd_r = analyze_multiple_allpass_sections(
                fr,
                spl_r,
                ph_r,
                max_sections=MAX_ALLPASS_SECTIONS,
                min_separation_hz=MIN_ALLPASS_SEPARATION_HZ,
            )

            src_ap_l.append(sections_l)
            src_ap_r.append(sections_r)

            should_apply = check_multiple_allpass_audibility(
                sections_l, sections_r, TARGET_FS
            )

            report_allpass_sections(
                az,
                el,
                sections_l,
                sections_r,
                debug_freqs_l,
                debug_gd_l,
                debug_freqs_r,
                debug_gd_r,
                visualizer,
                DEBUG,
            )

    src_pos = np.array(src_pos)
    src_mags_l = np.array(src_mags_l)
    src_mags_r = np.array(src_mags_r)

    visualizer.plot_source_magnitudes(
        src_pos, src_mags_l, src_mags_r, TARGET_FS, N_SAMPLES
    )

    visualizer.plot_symmetry_check(
        src_pos, src_mags_l, src_mags_r, TARGET_FS, N_SAMPLES
    )

    print("Calculating ITD...")
    final_pos = src_pos
    final_mags = np.stack([src_mags_l, src_mags_r], axis=-1)
    final_ap_l = src_ap_l
    final_ap_r = src_ap_r
    final_itds = calculate_woodworth_itd_array(src_pos, HEAD_WIDTH_CM, ITD_STRENGTH)

    visualizer.plot_itd_analysis(final_pos, final_itds)

    # Reconstruction
    final_mags = np.maximum(final_mags, 1e-9)
    print("Generating Minimum Phase HRIRs...")
    raw_hrirs = convert_to_min_phase(final_mags, N_SAMPLES, DEBUG)
    hrirs = np.transpose(raw_hrirs, (0, 2, 1))

    # Apply all-pass filters
    print("Applying All-Pass Filters...")

    for i in range(len(hrirs)):
        sections_l = final_ap_l[i] if i < len(final_ap_l) else []
        sections_r = final_ap_r[i] if i < len(final_ap_r) else []

        should_apply = check_multiple_allpass_audibility(
            sections_l, sections_r, TARGET_FS
        )

        if should_apply:
            hrirs[i, 0, :] = apply_multiple_allpass_filters(
                hrirs[i, 0, :], sections_l, TARGET_FS
            )
            hrirs[i, 1, :] = apply_multiple_allpass_filters(
                hrirs[i, 1, :], sections_r, TARGET_FS
            )

    # Apply ITD
    print("Applying ITD...")
    safety_offset = 5  # just in case

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

    # Fade out
    fade_out_len = N_SAMPLES // 4
    fade_out_window = np.hanning(2 * fade_out_len)[fade_out_len:]
    for i in range(len(hrirs)):
        for ch in range(2):
            hrirs[i, ch, -fade_out_len:] *= fade_out_window

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
