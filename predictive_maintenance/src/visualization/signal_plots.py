import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path


class SignalPlotter:

    def __init__(self, save_dir: str = "reports/figures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_machine_signals(
        self,
        machine_df: pd.DataFrame,
        machine_id: int,
        show_failures: bool = True,
        save: bool = True,
    ):
        df = machine_df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")

        sensor_cols = ["volt", "rotate", "pressure", "vibration"]
        fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
        fig.suptitle(f"Machine #{machine_id} — Sensor readings", fontsize=14)

        for ax, col in zip(axes, sensor_cols):
            ax.plot(df["datetime"], df[col], linewidth=0.5, color="steelblue", alpha=0.8)
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)

            if show_failures and "failure_component" in df.columns:
                failure_mask = df["failure_component"] != "none"
                if failure_mask.any():
                    ax.fill_between(
                        df["datetime"],
                        df[col].min(),
                        df[col].max(),
                        where=failure_mask,
                        color="red",
                        alpha=0.15,
                        label="Failure zone",
                    )

        axes[0].legend(loc="upper right")
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save:
            filepath = self.save_dir / f"machine_{machine_id}_signals.png"
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            print(f"Plot saved: {filepath}")

        plt.show()

    def plot_fft_comparison(
        self,
        healthy_window: pd.Series,
        failure_window: pd.Series,
        sensor_name: str = "vibration",
    ):
        from scipy.fft import fft, fftfreq

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        for ax, (title, window) in zip(
            axes,
            [("Healthy state", healthy_window), ("Pre-failure state", failure_window)],
        ):
            n = len(window)
            fft_vals = np.abs(fft(window.values))[:n//2]
            freqs = fftfreq(n, d=1/60)[:n//2]

            ax.plot(freqs, fft_vals, color="steelblue")
            ax.set_title(title)
            ax.set_xlabel("Frequency (1/min)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"Spectrum of {sensor_name} — state comparison")
        plt.tight_layout()
        plt.show()
