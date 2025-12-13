import numpy as np


def check_impulse_decay(impulse, n_fft, fade_len, hrir_idx, channel):
    """
    Check where impulse decays before fade out is applied
    """
    threshold = np.max(np.abs(impulse)) * 0.01  # 1% of peak
    decay_sample = n_fft
    for sample_idx in range(len(impulse) - 1, -1, -1):
        if np.abs(impulse[sample_idx]) > threshold:
            decay_sample = sample_idx
            break
    print(
        f"  HRIR[{hrir_idx}, ch{channel}]: impulse decays at sample {decay_sample}/{n_fft}, "
        f"fade starts at {n_fft - fade_len}"
    )


def convert_to_min_phase(magnitudes, n_fft, DEBUG=False):
    """
    Generates Minimum-Phase HRIRs from magnitude responses using the Cepstral method.

    1. Compute Log-Magnitude Spectrum.
    2. Compute Real Cepstrum via IFFT.
    3. Apply Cepstral Window to enforce Minimum Phase.
    4. Compute Complex Spectrum via FFT and Exponentiation.
    5. Compute Time-Domain Impulse (HRIR) via IFFT.
    6. Apply Hanning fade-out to the tail.

    Args:
        magnitudes: Array of magnitude responses. Shape: (num_hrirs, freq_bins, 2)
        n_fft: FFT size for the time-domain reconstruction
        DEBUG: If True, prints decay diagnostics for each filter

    Returns:
        hrirs: Time-domain Minimum-Phase HRIRs. Shape: (num_hrirs, n_fft, 2)
    """
    cep_window = np.zeros(n_fft)
    cep_window[0] = 1.0
    cep_window[1 : n_fft // 2] = 2.0
    cep_window[n_fft // 2] = 1.0

    fade_len = n_fft // 8  # 64 for 512 samples
    fade_window = np.hanning(2 * fade_len)[fade_len:]

    hrirs = np.zeros((magnitudes.shape[0], n_fft, 2))
    if DEBUG:
        print(10 * "-")

    for i in range(magnitudes.shape[0]):
        for ch in range(2):
            mag = magnitudes[i, :, ch]
            log_mag = np.log(np.maximum(mag, 1e-9))
            cepstrum = np.fft.irfft(log_mag, n=n_fft)
            cepstrum *= cep_window
            min_phase_spec = np.exp(np.fft.rfft(cepstrum, n=n_fft))
            impulse = np.fft.irfft(min_phase_spec, n=n_fft)

            if DEBUG:
                check_impulse_decay(impulse, n_fft, fade_len, i, ch)

            impulse[-fade_len:] *= fade_window

            hrirs[i, :, ch] = impulse
    if DEBUG:
        print(10 * "-")

    return hrirs
