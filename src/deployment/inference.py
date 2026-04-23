import time
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from ..models.lgbm_classifier import LGBMClassifier
from ..features.rolling_aggregator import RollingAggregator
from ..features.deep_embeddings import TimesNetEmbedder


class InferencePipeline:

    def __init__(
        self,
        model_path: str = "models/lgbm_model.joblib",
        feature_config_path: str = "configs/feature_config.yaml",
        alert_threshold: float = 0.3,
    ):
        self.model = LGBMClassifier.load(model_path)
        self.aggregator = RollingAggregator(feature_config_path)

        self.embedder: Optional[TimesNetEmbedder] = None
        import yaml
        with open(feature_config_path, "r") as f:
            feature_config = yaml.safe_load(f)
        if feature_config.get("timesnet", {}).get("enabled", False):
            timesnet_config = feature_config["timesnet"]
            self.embedder = TimesNetEmbedder(
                pretrained_path=timesnet_config.get("pretrained_path", ""),
                embedding_dim=timesnet_config.get("embedding_dim", 512),
                input_window_size=timesnet_config.get("input_window_size", 96),
            )

        self.alert_threshold = alert_threshold
        self.buffer: Dict[int, List[Dict]] = {}
        self.sensor_cols = ["volt", "rotate", "pressure", "vibration"]

    def ingest(self, machine_id: int, timestamp, measurements: Dict[str, float]) -> Optional[Dict]:
        start_time = time.perf_counter()

        if machine_id not in self.buffer:
            self.buffer[machine_id] = []

        self.buffer[machine_id].append({
            "datetime": timestamp,
            "machineID": machine_id,
            **measurements,
        })
        if len(self.buffer[machine_id]) > 200:
            self.buffer[machine_id].pop(0)

        if len(self.buffer[machine_id]) < 5:
            return None

        mini_df = pd.DataFrame(self.buffer[machine_id])

        features_df = self.aggregator.aggregate_machine(mini_df, self.sensor_cols)
        latest_features = features_df.iloc[-1:]

        if self.embedder is not None:
            window_data = mini_df[self.sensor_cols].values[-96:]
            emb_features = self.embedder.extract_embeddings(window_data)
            for k, v in emb_features.items():
                latest_features[k] = v

        drop_cols = ["datetime", "machineID", "failure_component"]
        feature_cols = [c for c in latest_features.columns if c not in drop_cols]
        X = latest_features[feature_cols]

        proba = self.model.predict_proba(X)
        if len(proba.shape) > 1:
            risk_score = 1 - proba[0][0]
            failure_type = np.argmax(proba[0][1:]) + 1 if risk_score >= self.alert_threshold else 0
        else:
            risk_score = proba[0]
            failure_type = 1 if risk_score >= self.alert_threshold else 0

        latency_ms = (time.perf_counter() - start_time) * 1000

        if risk_score >= self.alert_threshold:
            return {
                "timestamp": timestamp,
                "machine_id": machine_id,
                "risk_score": float(risk_score),
                "failure_type": int(failure_type),
                "alert": True,
                "latency_ms": latency_ms,
            }

        return {
            "timestamp": timestamp,
            "machine_id": machine_id,
            "risk_score": float(risk_score),
            "alert": False,
            "latency_ms": latency_ms,
        }

    def benchmark_latency(self, n_iterations: int = 1000) -> Dict[str, float]:
        latencies = []
        dummy_measurement = {
            "volt": 170.0,
            "rotate": 450.0,
            "pressure": 100.0,
            "vibration": 40.0,
        }

        for _ in range(n_iterations):
            start = time.perf_counter()
            self.ingest(1, pd.Timestamp.now(), dummy_measurement)
            latencies.append((time.perf_counter() - start) * 1000)

        return {
            "mean_ms": np.mean(latencies),
            "p50_ms": np.percentile(latencies, 50),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
            "max_ms": np.max(latencies),
        }
