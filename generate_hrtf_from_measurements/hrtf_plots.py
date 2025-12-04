import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import os


class HRTFVisualizer:
    """Class for visualizing all stages of HRTF processing"""

    def __init__(self, output_folder="hrtf_processing_plots"):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

    def _save_plot(self, filename):
        plt.tight_layout()
        filepath = os.path.join(self.output_folder, filename)
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"Saved: {filename}")

    def plot_loaded_data(self, data_map):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        azimuths = [az for (az, el) in data_map.keys()]
        elevations = [el for (az, el) in data_map.keys()]

        ax1.scatter(azimuths, elevations, alpha=0.6, s=50)
        ax1.set_xlabel("Azimuth (deg)")
        ax1.set_ylabel("Elevation (deg)")
        ax1.set_title(f"Loaded Measurements: {len(data_map)} points")
        ax1.grid(True, alpha=0.3)

        sample_key = list(data_map.keys())[0]
        if "left" in data_map[sample_key]:
            freq, spl = data_map[sample_key]["left"]
            ax2.semilogx(freq, spl, label="Left", alpha=0.7)
        if "right" in data_map[sample_key]:
            freq, spl = data_map[sample_key]["right"]
            ax2.semilogx(freq, spl, label="Right", alpha=0.7)

        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("SPL (dB)")
        ax2.set_title(
            f"Example: Az={sample_key[0]}deg, El={sample_key[1]}deg (RAW DATA)"
        )
        ax2.set_xlim([400, 22000])
        ax2.legend()
        ax2.grid(True, alpha=0.3, which="both")

        self._save_plot("01_loaded_data.png")

    def plot_trend_correction(self, data_map, fs, n_samples):
        """Visualization of balance correction"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        sample_pair = self._find_sample_pair(data_map)

        if sample_pair:
            (az1, el1), (az2, el2) = sample_pair

            freq1_l, spl1_l = data_map[(az1, el1)]["left"]
            freq1_r, spl1_r = data_map[(az1, el1)]["right"]

            axes[0, 0].semilogx(freq1_l, spl1_l, label="Left", linewidth=2)
            axes[0, 0].semilogx(freq1_r, spl1_r, label="Right", linewidth=2)
            axes[0, 0].set_title(f"Before Trend: Az={az1}deg")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].semilogx(freq1_l, spl1_l - spl1_r, color="red", linewidth=2)
            axes[0, 1].set_title("L-R Difference (Before)")
            axes[0, 1].set_ylabel("dB")
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axhline(0, color="black", linestyle="--", alpha=0.5)

            freq2_l, spl2_l = data_map[(az2, el2)]["left"]
            freq2_r, spl2_r = data_map[(az2, el2)]["right"]

            axes[1, 0].semilogx(freq2_l, spl2_l, label="Left", linewidth=2)
            axes[1, 0].semilogx(freq2_r, spl2_r, label="Right", linewidth=2)
            axes[1, 0].set_title(f"Before Trend (Mirror): Az={az2}deg")
            axes[1, 0].set_xlabel("Frequency (Hz)")
            axes[1, 0].set_ylabel("SPL (dB)")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].semilogx(freq2_l, spl2_l - spl2_r, color="red", linewidth=2)
            axes[1, 1].set_title("L-R Difference (Mirror, Before)")
            axes[1, 1].set_xlabel("Frequency (Hz)")
            axes[1, 1].set_ylabel("dB")
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(0, color="black", linestyle="--", alpha=0.5)

        self._save_plot("02_before_trend.png")

    def plot_after_trend(self, data_map):
        """Visualization after correction"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        sample_pair = self._find_sample_pair(data_map)

        if sample_pair:
            (az1, el1), (az2, el2) = sample_pair

            freq1_l, spl1_l = data_map[(az1, el1)]["left"]
            freq1_r, spl1_r = data_map[(az1, el1)]["right"]

            axes[0, 0].semilogx(freq1_l, spl1_l, label="Left", linewidth=2)
            axes[0, 0].semilogx(freq1_r, spl1_r, label="Right (corrected)", linewidth=2)
            axes[0, 0].set_title(f"After Trend: Az={az1}deg")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].semilogx(freq1_l, spl1_l - spl1_r, color="green", linewidth=2)
            axes[0, 1].set_title("L-R Difference (After)")
            axes[0, 1].set_ylabel("dB")
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axhline(0, color="black", linestyle="--", alpha=0.5)

            freq2_l, spl2_l = data_map[(az2, el2)]["left"]
            freq2_r, spl2_r = data_map[(az2, el2)]["right"]

            axes[1, 0].semilogx(freq2_l, spl2_l, label="Left", linewidth=2)
            axes[1, 0].semilogx(freq2_r, spl2_r, label="Right (corrected)", linewidth=2)
            axes[1, 0].set_title(f"Mirror Point: Az={az2}deg")
            axes[1, 0].set_xlabel("Frequency (Hz)")
            axes[1, 0].set_ylabel("SPL (dB)")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].semilogx(freq2_l, spl2_l - spl2_r, color="green", linewidth=2)
            axes[1, 1].set_title("L-R Difference (Mirror)")
            axes[1, 1].set_xlabel("Frequency (Hz)")
            axes[1, 1].set_ylabel("dB")
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(0, color="black", linestyle="--", alpha=0.5)

        self._save_plot("03_after_trend.png")

    def plot_source_magnitudes(
        self, src_pos, src_mags_l, src_mags_r, target_fs, n_samples
    ):
        """Visualization of the original spectra"""
        freqs = np.fft.rfftfreq(n_samples, 1.0 / target_fs)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        avg_l_mag = np.mean(src_mags_l, axis=0)
        avg_r_mag = np.mean(src_mags_r, axis=0)

        axes[0, 0].semilogx(
            freqs, 20 * np.log10(avg_l_mag + 1e-9), label="Left", linewidth=2
        )
        axes[0, 0].semilogx(
            freqs, 20 * np.log10(avg_r_mag + 1e-9), label="Right", linewidth=2
        )
        axes[0, 0].set_xlabel("Frequency (Hz)")
        axes[0, 0].set_ylabel("Magnitude (dB)")
        axes[0, 0].set_title("Average Magnitude Response")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        test_bins = [int(f * n_samples / target_fs) for f in [1000, 5000, 10000]]
        test_labels = ["1kHz", "5kHz", "10kHz"]

        for bin_idx, label in zip(test_bins, test_labels):
            values_l = 20 * np.log10(src_mags_l[:, bin_idx] + 1e-9)
            axes[0, 1].hist(values_l, bins=20, alpha=0.5, label=label)

        axes[0, 1].set_xlabel("Magnitude (dB)")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title("Distribution at Key Frequencies (Left)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        test_indices = [0, len(src_pos) // 4, len(src_pos) // 2, 3 * len(src_pos) // 4]
        for idx in test_indices:
            if idx < len(src_pos):
                az, el = src_pos[idx, 0], src_pos[idx, 1]
                axes[1, 0].semilogx(
                    freqs,
                    20 * np.log10(src_mags_l[idx] + 1e-9),
                    alpha=0.7,
                    label=f"Az={az:.0f}deg El={el:.0f}deg",
                )

        axes[1, 0].set_xlabel("Frequency (Hz)")
        axes[1, 0].set_ylabel("Magnitude (dB)")
        axes[1, 0].set_title("Sample Source Spectra (Left)")
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

        scatter = axes[1, 1].scatter(
            src_pos[:, 0],
            src_pos[:, 1],
            c=20 * np.log10(np.mean(src_mags_l, axis=1)),
            cmap="viridis",
            s=50,
        )
        axes[1, 1].set_xlabel("Azimuth (deg)")
        axes[1, 1].set_ylabel("Elevation (deg)")
        axes[1, 1].set_title("Spatial Distribution (colored by avg magnitude)")
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label("Avg Magnitude (dB)")
        axes[1, 1].grid(True, alpha=0.3)

        self._save_plot("04_source_magnitudes.png")

    def plot_interpolation_comparison(
        self, src_pos, target_pos, src_mags_l, interp_mags_l, target_fs, n_samples
    ):
        """Comparison before and after interpolation"""
        freqs = np.fft.rfftfreq(n_samples, 1.0 / target_fs)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].scatter(
            src_pos[:, 0], src_pos[:, 1], s=50, alpha=0.6, label="Source", color="red"
        )
        axes[0, 0].scatter(
            target_pos[:, 0],
            target_pos[:, 1],
            s=10,
            alpha=0.3,
            label="Target Grid",
            color="blue",
        )
        axes[0, 0].set_xlabel("Azimuth (deg)")
        axes[0, 0].set_ylabel("Elevation (deg)")
        axes[0, 0].set_title(f"Coverage: {len(src_pos)} → {len(target_pos)} points")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        test_target_idx = len(target_pos) // 3
        test_az, test_el = (
            target_pos[test_target_idx, 0],
            target_pos[test_target_idx, 1],
        )

        distances = np.sqrt(
            (src_pos[:, 0] - test_az) ** 2 + (src_pos[:, 1] - test_el) ** 2
        )
        nearest_src_idx = np.argmin(distances)

        axes[0, 1].semilogx(
            freqs,
            20 * np.log10(src_mags_l[nearest_src_idx] + 1e-9),
            label=f"Nearest Source ({distances[nearest_src_idx]:.1f}deg away)",
            linewidth=2,
        )
        axes[0, 1].semilogx(
            freqs,
            20 * np.log10(interp_mags_l[test_target_idx] + 1e-9),
            label="Interpolated",
            linewidth=2,
            linestyle="--",
        )
        axes[0, 1].set_xlabel("Frequency (Hz)")
        axes[0, 1].set_ylabel("Magnitude (dB)")
        axes[0, 1].set_title(f"Test Point: Az={test_az:.1f}deg El={test_el:.1f}deg")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        avg_src = np.mean(src_mags_l, axis=0)
        avg_interp = np.mean(interp_mags_l, axis=0)

        axes[1, 0].semilogx(
            freqs, 20 * np.log10(avg_src + 1e-9), label="Source Average", linewidth=2
        )
        axes[1, 0].semilogx(
            freqs,
            20 * np.log10(avg_interp + 1e-9),
            label="Interpolated Average",
            linewidth=2,
            linestyle="--",
        )
        axes[1, 0].set_xlabel("Frequency (Hz)")
        axes[1, 0].set_ylabel("Magnitude (dB)")
        axes[1, 0].set_title("Average Spectral Comparison")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].semilogx(
            freqs,
            20 * np.log10(avg_interp + 1e-9) - 20 * np.log10(avg_src + 1e-9),
            color="green",
            linewidth=2,
        )
        axes[1, 1].set_xlabel("Frequency (Hz)")
        axes[1, 1].set_ylabel("Difference (dB)")
        axes[1, 1].set_title("Interpolation Error (Avg)")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(0, color="black", linestyle="--", alpha=0.5)

        self._save_plot("05_interpolation.png")

    def plot_itd_analysis(self, final_pos, final_itds):
        """Visualization of ITD"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        scatter = axes[0, 0].scatter(
            final_pos[:, 0],
            final_pos[:, 1],
            c=final_itds * 1000,
            cmap="RdBu_r",
            s=30,
            vmin=-0.8,
            vmax=0.8,
        )
        axes[0, 0].set_xlabel("Azimuth (deg)")
        axes[0, 0].set_ylabel("Elevation (deg)")
        axes[0, 0].set_title("ITD Spatial Distribution")
        cbar = plt.colorbar(scatter, ax=axes[0, 0])
        cbar.set_label("ITD (ms)")
        axes[0, 0].grid(True, alpha=0.3)

        equator_mask = np.abs(final_pos[:, 1]) < 5
        equator_az = final_pos[equator_mask, 0]
        equator_itd = final_itds[equator_mask] * 1000

        sort_idx = np.argsort(equator_az)
        axes[0, 1].plot(
            equator_az[sort_idx], equator_itd[sort_idx], "o-", linewidth=2, markersize=4
        )
        axes[0, 1].set_xlabel("Azimuth (deg)")
        axes[0, 1].set_ylabel("ITD (ms)")
        axes[0, 1].set_title("ITD vs Azimuth (Elevation ≈ 0deg)")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(0, color="black", linestyle="--", alpha=0.5)

        front_mask = (np.abs(final_pos[:, 0]) < 5) | (np.abs(final_pos[:, 0] - 360) < 5)
        front_el = final_pos[front_mask, 1]
        front_itd = final_itds[front_mask] * 1000

        sort_idx = np.argsort(front_el)
        axes[1, 0].plot(
            front_el[sort_idx], front_itd[sort_idx], "o-", linewidth=2, markersize=4
        )
        axes[1, 0].set_xlabel("Elevation (deg)")
        axes[1, 0].set_ylabel("ITD (ms)")
        axes[1, 0].set_title("ITD vs Elevation (Azimuth ≈ 0deg)")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(0, color="black", linestyle="--", alpha=0.5)

        axes[1, 1].hist(final_itds * 1000, bins=50, edgecolor="black", alpha=0.7)
        axes[1, 1].set_xlabel("ITD (ms)")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title(
            f"ITD Distribution (Range: {final_itds.min()*1000:.2f} to {final_itds.max()*1000:.2f} ms)"
        )
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(0, color="red", linestyle="--", linewidth=2, label="Zero")
        axes[1, 1].legend()

        self._save_plot("06_itd_analysis.png")

    def plot_final_hrirs(self, hrirs, final_pos, target_fs):
        """Final"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        test_directions = [
            (0, 0, "Front"),
            (90, 0, "Right"),
            (180, 0, "Back"),
            (270, 0, "Left"),
            (45, 30, "Front-Right Up"),
        ]

        for i, (az, el, label) in enumerate(test_directions):
            if i >= 5:
                break
            distances = np.sqrt(
                (final_pos[:, 0] - az) ** 2 + (final_pos[:, 1] - el) ** 2
            )
            idx = np.argmin(distances)

            time_axis = np.arange(hrirs.shape[2]) / target_fs * 1000

            row = i // 3
            col = i % 3

            axes[row, col].plot(time_axis, hrirs[idx, 0, :], label="Left", linewidth=1)
            axes[row, col].plot(
                time_axis, hrirs[idx, 1, :], label="Right", linewidth=1, alpha=0.7
            )
            axes[row, col].set_xlabel("Time (ms)")
            axes[row, col].set_ylabel("Amplitude")
            axes[row, col].set_title(
                f"{label} (Az={final_pos[idx,0]:.0f}deg, El={final_pos[idx,1]:.0f}deg)"
            )
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)

        test_idx = np.argmin(
            np.sqrt((final_pos[:, 0] - 90) ** 2 + final_pos[:, 1] ** 2)
        )
        from scipy import signal

        f, t, Sxx = signal.spectrogram(hrirs[test_idx, 0, :], target_fs, nperseg=256)

        axes[1, 2].pcolormesh(
            t * 1000, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud", cmap="viridis"
        )
        axes[1, 2].set_ylabel("Frequency (Hz)")
        axes[1, 2].set_xlabel("Time (ms)")
        axes[1, 2].set_title("Spectrogram (Right, Left channel)")
        axes[1, 2].set_ylim([0, 20000])

        self._save_plot("07_final_hrirs.png")

    def _find_sample_pair(self, data_map):
        """Helper function to find a pair of measurements"""
        for el in [0, -10, 10]:
            for az in [30, 45, 60]:
                mirror_az = (360 - az) % 360
                if (az, el) in data_map and (mirror_az, el) in data_map:
                    return ((az, el), (mirror_az, el))
        return None

    def plot_symmetry_check(
        self, src_pos, src_mags_l, src_mags_r, target_fs, n_samples
    ):
        """
        Check symmetry and flat shelf for specific angles on the equator (El=0).
        Groups: 0, (30-330), (90-270), (150-210), 180
        """
        freqs = np.fft.rfftfreq(n_samples, 1.0 / target_fs)

        groups = [
            ([0], "Front (0deg)"),
            ([30, 330], "Front-Side (30deg & 330deg)"),
            ([90, 270], "Side (90deg & 270deg)"),
            ([150, 210], "Back-Side (150deg & 210deg)"),
            ([180], "Back (180deg)"),
        ]

        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        axes = axes.flatten()

        for i, (az_list, title) in enumerate(groups):
            ax = axes[i]

            has_data = False
            for az_target in az_list:
                indices = np.where(
                    (np.abs(src_pos[:, 0] - az_target) < 1.5)
                    & (np.abs(src_pos[:, 1]) < 1.5)
                )[0]

                if len(indices) > 0:
                    idx = indices[0]
                    has_data = True

                    mag_l = 20 * np.log10(src_mags_l[idx] + 1e-9)
                    mag_r = 20 * np.log10(src_mags_r[idx] + 1e-9)

                    if az_target in [0, 180, 30, 90, 150]:
                        style_l, style_r = "-", "--"
                        color_base = "tab:blue"
                    else:
                        style_l, style_r = "-", "--"
                        color_base = "tab:orange"

                    ax.semilogx(
                        freqs,
                        mag_l,
                        label=f"Az={az_target}deg Left",
                        linestyle=style_l,
                        linewidth=2,
                    )
                    ax.semilogx(
                        freqs,
                        mag_r,
                        label=f"Az={az_target}deg Right",
                        linestyle=style_r,
                        linewidth=2,
                    )

            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude (dB)")
            ax.grid(True, which="both", alpha=0.3)
            ax.set_xlim([20, 22000])

            if has_data:
                ax.legend(fontsize=9, loc="best")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No Data Found for El=0",
                    ha="center",
                    transform=ax.transAxes,
                )

        fig.delaxes(axes[5])

        self._save_plot("05_symmetry_check.png")
