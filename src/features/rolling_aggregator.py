import numpy as np
import pandas as pd
import yaml
from typing import Dict, List
from .stat_features import StatFeatureExtractor
from .spectral_features import SpectralFeatureExtractor


class RollingAggregator:

    def __init__(self, feature_config_path: str = "configs/feature_config.yaml"):
        with open(feature_config_path, "r") as f:
            self.feature_config = yaml.safe_load(f)

        self.stat_extractor = StatFeatureExtractor(
            functions=self.feature_config["statistical"]["functions"]
        )
        self.spectral_extractor = SpectralFeatureExtractor(
            n_components=self.feature_config["spectral"]["fft_components"]
        )
        self.windows = {
            "5min": 5,
            "30min": 30,
            "2h": 120,
        }

    def aggregate_machine(
        self, machine_df: pd.DataFrame, sensor_cols: List[str]
    ) -> pd.DataFrame:
        result_rows = []

        for idx in range(len(machine_df)):
            row_features = {
                "datetime": machine_df.iloc[idx]["datetime"],
                "machineID": machine_df.iloc[idx]["machineID"],
                "failure_component": machine_df.iloc[idx]["failure_component"],
            }

            for window_name, window_size in self.windows.items():
                start_idx = max(0, idx - window_size + 1)
                window = machine_df.iloc[start_idx : idx + 1]

                for col in sensor_cols:
                    window_data = window[col].values
                    prefix = f"{col}_{window_name}_"

                    row_features.update(
                        self.stat_extractor.extract(window_data, prefix)
                    )
                    if col == "vibration":
                        row_features.update(
                            self.spectral_extractor.extract_all(window_data, prefix)
                        )

            result_rows.append(row_features)

        return pd.DataFrame(result_rows)
