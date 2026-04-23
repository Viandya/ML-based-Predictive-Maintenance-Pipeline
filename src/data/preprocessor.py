import numpy as np
import pandas as pd
import yaml
from typing import Tuple


class DataPreprocessor:

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.labeling = self.config["labeling"]

    def clean_telemetry(self, df: pd.DataFrame) -> pd.DataFrame:
        sensor_cols = ["volt", "rotate", "pressure", "vibration"]

        for col in sensor_cols:
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = (df[col] < lower) | (df[col] > upper)
            n_outliers = outliers.sum()
            df.loc[outliers, col] = np.nan
            df[col] = df[col].interpolate(method="linear", limit=10)

            print(f"  {col}: заменено {n_outliers:,} выбросов ({n_outliers/len(df)*100:.2f}%)")

        df[sensor_cols] = df[sensor_cols].interpolate(method="linear", limit_direction="both")

        return df

    def merge_errors(
        self, telemetry: pd.DataFrame, errors: pd.DataFrame
    ) -> pd.DataFrame:
        errors["datetime"] = pd.to_datetime(errors["datetime"])
        telemetry = telemetry.copy()
        telemetry["datetime"] = pd.to_datetime(telemetry["datetime"])

        telemetry["failure_component"] = "none"

        for machine_id in telemetry["machineID"].unique():
            machine_errors = errors[errors["machineID"] == machine_id].sort_values("datetime")
            machine_telemetry = telemetry[telemetry["machineID"] == machine_id].copy()

            for _, error_row in machine_errors.iterrows():
                error_time = error_row["datetime"]
                failure_window_days = self.labeling["failure_window_days"]
                alert_horizon = pd.Timedelta(hours=self.labeling["alert_horizon_hours"])

                mask = (
                    (telemetry["machineID"] == machine_id)
                    & (telemetry["datetime"] < error_time)
                    & (telemetry["datetime"] >= error_time - pd.Timedelta(days=failure_window_days))
                )
                telemetry.loc[mask, "failure_component"] = error_row["errorID"]

                healthy_mask = (
                    (telemetry["machineID"] == machine_id)
                    & (telemetry["datetime"] >= error_time)
                    & (telemetry["datetime"] < error_time + pd.Timedelta(days=self.labeling["healthy_window_days"]))
                )
                telemetry.loc[healthy_mask, "failure_component"] = "none"

        print(f"Размечено отказов: {(telemetry['failure_component'] != 'none').sum():,}")
        print(f"Распределение классов:\n{telemetry['failure_component'].value_counts()}")

        return telemetry

    def prepare_features_target(
        self, telemetry: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        feature_cols = ["volt", "rotate", "pressure", "vibration"]
        X = telemetry[feature_cols]
        y = telemetry["failure_component"]
        return X, y
