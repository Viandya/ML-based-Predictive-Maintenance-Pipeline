import numpy as np
from scipy import stats
from typing import List, Dict


class StatFeatureExtractor:

    def __init__(self, functions: List[str] = None):
        self.functions = functions or ["mean", "std", "min", "max", "kurtosis", "skew", "q25", "q75"]
        self._func_map = {
            "mean": np.mean,
            "std": np.std,
            "min": np.min,
            "max": np.max,
            "kurtosis": stats.kurtosis,
            "skew": stats.skew,
            "q25": lambda x: np.percentile(x, 25),
            "q75": lambda x: np.percentile(x, 75),
        }

    def extract(self, window_data: np.ndarray, prefix: str = "") -> Dict[str, float]:
        features = {}
        for func_name in self.functions:
            try:
                value = self._func_map[func_name](window_data)
                if np.isnan(value):
                    value = 0.0
                features[f"{prefix}{func_name}"] = float(value)
            except Exception:
                features[f"{prefix}{func_name}"] = 0.0
        return features
