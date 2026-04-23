from pathlib import Path
import pandas as pd
import yaml
from typing import Tuple


class DataLoader:

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.raw_path = Path(self.config["paths"]["raw_data"])

    def load_telemetry(self) -> pd.DataFrame:
        file_path = self.raw_path / self.config["data"]["telemetry_file"]
        df = pd.read_csv(file_path, parse_dates=["datetime"])
        df = df.sort_values(["machineID", "datetime"])
        print(f"Загружена телеметрия: {df.shape[0]:,} записей, {df['machineID'].nunique()} станков")
        return df

    def load_errors(self) -> pd.DataFrame:
        file_path = self.raw_path / self.config["data"]["errors_file"]
        df = pd.read_csv(file_path, parse_dates=["datetime"])
        print(f"Загружен журнал ошибок: {df.shape[0]:,} записей")
        return df

    def load_machines(self) -> pd.DataFrame:
        file_path = self.raw_path / self.config["data"]["machines_file"]
        df = pd.read_csv(file_path)
        print(f"Загружены метаданные: {df.shape[0]} станков")
        return df

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.load_telemetry(), self.load_errors(), self.load_machines()
