import numpy as np
import pandas as pd


class DataAugmenter:

    def __init__(self, noise_level: float = 0.02, random_state: int = 42):
        self.noise_level = noise_level
        self.rng = np.random.RandomState(random_state)

    def add_sensor_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        sensor_cols = ["volt", "rotate", "pressure", "vibration"]
        df_noisy = df.copy()

        for col in sensor_cols:
            std = df[col].std()
            noise = self.rng.normal(0, std * self.noise_level, size=len(df))
            df_noisy[col] = df[col] + noise

        print(f"Добавлен шум к датчикам (уровень: {self.noise_level})")
        return df_noisy

    def add_random_walk_drift(self, df: pd.DataFrame, drift_std: float = 0.001) -> pd.DataFrame:
        sensor_cols = ["volt", "rotate", "pressure", "vibration"]
        df_drifted = df.copy()

        for machine_id in df["machineID"].unique():
            mask = df["machineID"] == machine_id
            n_points = mask.sum()

            for col in sensor_cols:
                steps = self.rng.normal(0, drift_std, size=n_points)
                drift = np.cumsum(steps)
                df_drifted.loc[mask, col] = df.loc[mask, col] + drift

        print(f"Добавлен сенсорный дрейф (random walk)")
        return df_drifted

    def undersample_healthy(self, df: pd.DataFrame, target_ratio: float = 0.001) -> pd.DataFrame:
        healthy_mask = df["failure_component"] == "none"
        failure_mask = df["failure_component"] != "none"

        n_failures = failure_mask.sum()
        n_healthy_keep = int(n_failures / target_ratio)

        healthy_indices = df[healthy_mask].index
        keep_indices = self.rng.choice(healthy_indices, size=min(n_healthy_keep, len(healthy_indices)), replace=False)
        drop_indices = healthy_indices.difference(keep_indices)

        df_imbalanced = df.drop(drop_indices)
        print(f"Усилен дисбаланс: было {healthy_mask.sum():,} здоровых, стало {len(keep_indices):,} (1:{1/target_ratio:.0f})")

        return df_imbalanced
