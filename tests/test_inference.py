import pytest
import numpy as np
import pandas as pd
from src.deployment.inference import InferencePipeline


class TestInferencePipeline:

    @pytest.fixture
    def pipeline(self, tmp_path):
        from src.models.lgbm_classifier import LGBMClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=100, n_features=50, random_state=42)
        X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(50)])
        y = pd.Series(y)

        model = LGBMClassifier()
        model.lgbm_params["n_estimators"] = 10
        model.fit(X, y)

        model_path = tmp_path / "test_model.joblib"
        model.save(str(model_path))

        pipeline = InferencePipeline(
            model_path=str(model_path),
            feature_config_path="configs/feature_config.yaml",
            alert_threshold=0.5,
        )
        pipeline.embedder = None
        return pipeline

    def test_ingest_normal(self, pipeline):
        measurements = {
            "volt": 170.0,
            "rotate": 450.0,
            "pressure": 100.0,
            "vibration": 40.0,
        }

        for _ in range(10):
            result = pipeline.ingest(1, pd.Timestamp.now(), measurements)

        assert result is not None
        assert "risk_score" in result
        assert "alert" in result
        assert "latency_ms" in result

    def test_latency_benchmark(self, pipeline):
        stats = pipeline.benchmark_latency(n_iterations=10)
        assert stats["mean_ms"] > 0
        assert stats["max_ms"] < 1000
