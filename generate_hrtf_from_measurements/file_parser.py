import os
import numpy as np
import re


def parse_rew_file(filepath, freq_start, freq_end):
    freqs, spl, phase = [], [], []
    if not os.path.exists(filepath):
        return np.array([]), np.array([]), np.array([])

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("*"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    f_val = float(parts[0])
                    s_val = float(parts[1])
                    p_val = float(parts[2])
                    if freq_start <= f_val <= freq_end:
                        freqs.append(f_val)
                        spl.append(s_val)
                        phase.append(p_val)
                except ValueError:
                    continue
    return np.array(freqs), np.array(spl), np.array(phase)


def parse_filename(filename):
    match = re.match(r"(\d+)_(left|right)(?:_([+-]?\d+)_elevation)?\.txt", filename)
    if match:
        return (
            int(match.group(1)),
            int(match.group(3)) if match.group(3) else 0,
            match.group(2),
        )
    return None, None, None
