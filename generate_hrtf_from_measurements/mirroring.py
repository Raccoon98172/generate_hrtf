import numpy as np


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
                f_l, s_l, p_l = data["left"]
                f_r, s_r, p_r = data["right"]

                s_l, s_r = blend_arrays(f_l, s_l, s_r)
                data["left"] = (f_l, s_l, p_l)
                data["right"] = (f_r, s_r, p_r)

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

                f_m, s_mL, p_mL = d_main["left"]
                f_mr, s_mR, p_mR = d_mirror["right"]
                s_mL, s_mR = blend_arrays(f_m, s_mL, s_mR)
                d_main["left"] = (f_m, s_mL, p_mL)
                d_mirror["right"] = (f_mr, s_mR, p_mR)

                f_m, s_mR, p_mR = d_main["right"]
                f_mr, s_mL, p_mL = d_mirror["left"]
                s_mR, s_mL = blend_arrays(f_m, s_mR, s_mL)
                d_main["right"] = (f_m, s_mR, p_mR)
                d_mirror["left"] = (f_mr, s_mL, p_mL)

                processed_pairs.add((az, el))
                processed_pairs.add((mirror_az, el))

    return data_map
