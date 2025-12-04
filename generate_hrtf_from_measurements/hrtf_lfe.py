import numpy as np


def _smoothstep_weights(x: np.ndarray) -> np.ndarray:
    """
    Creates an S-shaped weight curve from 0 to 1 using a cosine function.
    Ensures smooth touch (zero derivative) at the edges.

    Args:
        x (np.ndarray): Normalized values from 0 to 1.

    Returns:
        np.ndarray: Weights from 0 to 1.
    """
    return 0.5 - 0.5 * np.cos(x * np.pi)


def extend_low_frequencies(
    left_measured_freqs: np.ndarray,
    left_measured_spl: np.ndarray,
    right_measured_freqs: np.ndarray,
    right_measured_spl: np.ndarray,
    target_freqs: np.ndarray,
    crossover_freq: float = 550.0,
    transition_width_hz: float = 150.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstructs the low-frequency part of the HRTF.

    Args:
        left_measured_freqs, left_measured_spl: Measurements for the left ear.
        right_measured_freqs, right_measured_spl: Measurements for the right ear.
        target_freqs: Target frequency grid for the full spectrum.
        crossover_freq: Center frequency of the crossfade.
        transition_width_hz: Width of the transition zone.

    Returns:
        tuple[np.ndarray, np.ndarray]: (full_spectrum_left_dB, full_spectrum_right_dB).
    """
    if len(left_measured_freqs) == 0 or len(right_measured_freqs) == 0:
        zeros = np.zeros_like(target_freqs)
        return zeros, zeros

    left_anchor_spl = np.interp(crossover_freq, left_measured_freqs, left_measured_spl)
    right_anchor_spl = np.interp(
        crossover_freq, right_measured_freqs, right_measured_spl
    )
    common_anchor_spl = (left_anchor_spl + right_anchor_spl) / 2.0

    def _create_full_spectrum(measured_freqs, measured_spl, anchor_spl):
        full_range_spl = np.zeros_like(target_freqs)
        transition_start = crossover_freq - (transition_width_hz / 2.0)
        transition_end = crossover_freq + (transition_width_hz / 2.0)

        low_freq_mask = target_freqs < transition_start
        transition_mask = (target_freqs >= transition_start) & (
            target_freqs <= transition_end
        )
        high_freq_mask = target_freqs > transition_end

        full_range_spl[low_freq_mask] = anchor_spl

        interp_high_spl = np.interp(
            target_freqs[high_freq_mask],
            measured_freqs,
            measured_spl,
            right=measured_spl[-1],
        )
        full_range_spl[high_freq_mask] = interp_high_spl

        if np.any(transition_mask):
            trans_freqs = target_freqs[transition_mask]
            model_vals = np.full_like(trans_freqs, anchor_spl)
            measured_vals = np.interp(trans_freqs, measured_freqs, measured_spl)

            normalized_freqs = (trans_freqs - transition_start) / transition_width_hz

            weights = _smoothstep_weights(normalized_freqs)

            blended_spl = model_vals * (1.0 - weights) + measured_vals * weights
            full_range_spl[transition_mask] = blended_spl

        full_range_spl[0] = anchor_spl
        return full_range_spl

    final_left_spl = _create_full_spectrum(
        left_measured_freqs, left_measured_spl, common_anchor_spl
    )
    final_right_spl = _create_full_spectrum(
        right_measured_freqs, right_measured_spl, common_anchor_spl
    )

    return final_left_spl, final_right_spl
