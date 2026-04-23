"""
Script to download Predictive Maintenance dataset.
Uses Microsoft Azure open dataset for demonstration.
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def generate_synthetic_data(output_dir: str = "data/raw"):
    """
    Generate synthetic dataset for demonstration.
    
    Simulates 10 machines, 4 sensors, 1 year of data with 5-minute intervals.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    n_machines = 10
    start_date = pd.Timestamp("2023-01-01")
    end_date = pd.Timestamp("2023-12-31")
    freq = "5min"

    date_range = pd.date_range(start_date, end_date, freq=freq)
    print(f"Generating {len(date_range):,} timestamps for {n_machines} machines...")

    rows = []
    for machine_id in range(1, n_machines + 1):
        n = len(date_range)
        base_volt = 170 + np.random.normal(0, 2)
        base_rotate = 450 + np.random.normal(0, 10)
        base_pressure = 100 + np.random.normal(0, 3)
        base_vibration = 40 + np.random.normal(0, 2)

        time_idx = np.arange(n)
        trend = time_idx / (n * 2)

        volt = base_volt + trend * 2 + np.random.normal(0, 1, n)
        rotate = base_rotate + trend * 5 + np.random.normal(0, 3, n)
        pressure = base_pressure + trend * 1 + np.random.normal(0, 0.8, n)
        vibration = base_vibration + trend * 3 + np.random.normal(0, 1.5, n)

        failure_indices = np.random.choice(n, size=np.random.randint(3, 8), replace=False)
        for idx in failure_indices:
            window_size = np.random.randint(100, 500)
            start = max(0, idx - window_size)
            
            vibration[start:idx] += np.exp(np.linspace(0, 3, idx - start)) * 5
            pressure[start:idx] -= np.linspace(0, 15, idx - start)
            rotate[start:idx] += np.linspace(0, 30, idx - start)

        machine_df = pd.DataFrame({
            "datetime": date_range,
            "machineID": machine_id,
            "volt": volt,
            "rotate": rotate,
            "pressure": pressure,
            "vibration": vibration,
        })
        rows.append(machine_df)

    telemetry = pd.concat(rows, ignore_index=True)
    telemetry.to_csv(output_dir / "telemetry.csv", index=False)
    print(f"Telemetry saved: {output_dir / 'telemetry.csv'} ({telemetry.memory_usage(deep=True).sum() / 1024**2:.1f} MB)")

    errors = []
    for machine_id in range(1, n_machines + 1):
        machine_data = telemetry[telemetry["machineID"] == machine_id]
        vibration_spikes = machine_data[
            machine_data["vibration"] > machine_data["vibration"].quantile(0.995)
        ]
        
        error_dates = vibration_spikes.groupby(
            pd.Grouper(key="datetime", freq="7D")
        ).first().dropna()

        for _, row in error_dates.iterrows():
            errors.append({
                "datetime": row["datetime"],
                "machineID": machine_id,
                "errorID": f"comp{np.random.randint(1, 6)}",
            })

    errors_df = pd.DataFrame(errors)
    errors_df.to_csv(output_dir / "errors.csv", index=False)
    print(f"Error log saved: {output_dir / 'errors.csv'} ({len(errors_df)} records)")

    machines = pd.DataFrame({
        "machineID": range(1, n_machines + 1),
        "model": np.random.choice(["CNC-100", "CNC-200", "CNC-300"], n_machines),
        "age_years": np.random.randint(1, 10, n_machines),
    })
    machines.to_csv(output_dir / "machines.csv", index=False)
    print(f"Metadata saved: {output_dir / 'machines.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare dataset")
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory to save data",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("GENERATING SYNTHETIC DATASET")
    print("=" * 50)

    generate_synthetic_data(args.output_dir)

    print("\nDone! Data generated in", args.output_dir)


if __name__ == "__main__":
    main()
