import pytest
import numpy as np
import pandas as pd
from src.features.stat_features import StatFeatureExtractor
from src.features.spectral_features import SpectralFeatureExtractor
from src.features.rolling_aggregator import RollingAggregator


class TestStatFeatureExtractor:

    def test_extract_basic(self):
        extractor = StatFeatureExtractor(functions=["mean", "std"])
        window = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        features = extractor.extract(window, prefix="test_")

        assert "test_mean" in features
        assert "test_std" in features
        assert abs(features["test_mean"] - 3.0) < 0.01
        assert features["test_std"] > 0

    def test_extract_empty_window(self):
        extractor = StatFeatureExtractor()
        window = np.array([])
        features = extractor.extract(window, prefix="test_")
        assert "test_mean" in features

    def test_extract_single_value(self):
        extractor = StatFeatureExtractor(functions=["mean", "std", "kurtosis"])
        window = np.array([42.0])
        features = extractor.extract(window, prefix="test_")
        assert features["test_mean"] == 42.0
        assert features["test_std"] == 0.0


class TestSpectralFeatureExtractor:

    def test_fft_features(self):
        extractor = SpectralFeatureExtractor(n_components=5)
        t = np.linspace(0, 10, 1000)
        window = np.sin(2 * np.pi * 2 * t)
        features = extractor.extract_fft_features(window, prefix="test_")

        assert "test_fft_freq_0" in features
        assert "test_fft_amp_0" in features
        assert abs(features["test_fft_freq_0"] - 2.0) < 1.0

    def test_psd_features(self):
        extractor = SpectralFeatureExtractor()
        window = np.random.normal(0, 1, 500)
        features = extractor.extract_psd_features(window, prefix="test_")

        assert "test_psd_max" in features
        assert "test_psd_mean" in features
        assert "test_psd_total" in features
        assert features["test_psd_total"] > 0


class TestRollingAggregator:

    def test_aggregate_basic(self):
        aggregator = RollingAggregator()

        dates = pd.date_range("2023-01-01", periods=10, freq="5min")
        df = pd.DataFrame({
            "datetime": dates,
            "machineID": 1,
            "volt": np.random.normal(170, 1, 10),
            "rotate": np.random.normal(450, 5, 10),
            "pressure": np.random.normal(100, 2, 10),
            "vibration": np.random.normal(40, 1, 10),
            "failure_component": "none",
        })

        sensor_cols = ["volt", "rotate", "pressure", "vibration"]
        result = aggregator.aggregate_machine(df, sensor_cols)

        assert len(result) == len(df)
        assert any("mean" in c for c in result.columns)
        assert any("fft" in c for c in result.columns)
