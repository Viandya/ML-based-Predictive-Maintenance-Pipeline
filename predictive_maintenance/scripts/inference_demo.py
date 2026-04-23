"""
Real-time inference demonstration.
Run: python scripts/inference_demo.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.deployment.inference import InferencePipeline


def main():
    print("REAL-TIME INFERENCE DEMONSTRATION")

    print("\nInitializing InferencePipeline...")
    pipeline = InferencePipeline(
        model_path="models/lgbm_model.joblib",
        alert_threshold=0.3,
    )

    print("\nMeasuring latency...")
    latency_stats = pipeline.benchmark_latency(n_iterations=500)
    print(f"  Mean latency: {latency_stats['mean_ms']:.2f} ms")
    print(f"  P95 latency:  {latency_stats['p95_ms']:.2f} ms")
    print(f"  P99 latency:  {latency_stats['p99_ms']:.2f} ms")
    print(f"  Max latency:  {latency_stats['max_ms']:.2f} ms")

    print("\nSimulating data stream (20 measurements)...")
    print("-" * 60)

    np.random.seed(123)
    base_measurements = {
        "volt": 170.0,
        "rotate": 450.0,
        "pressure": 100.0,
        "vibration": 40.0,
    }

    alert_count = 0
    for i in range(20):
        measurements = {
            k: v + np.random.normal(0, abs(v) * 0.01)
            for k, v in base_measurements.items()
        }

        if i >= 15:
            measurements["vibration"] += (i - 14) * 5
            measurements["pressure"] -= (i - 14) * 2

        result = pipeline.ingest(
            machine_id=1,
            timestamp=pd.Timestamp.now(),
            measurements=measurements,
        )

        if result:
            status = "ALERT!" if result["alert"] else "Norm"
            print(f"  [{result['timestamp'].strftime('%H:%M:%S')}] {status} "
                  f"risk={result['risk_score']:.3f} "
                  f"latency={result['latency_ms']:.2f}ms", end="")
            
            if result["alert"]:
                alert_count += 1
                print(f" failure_type={result['failure_type']}")
            else:
                print()

        time.sleep(0.1)
    print(f"Total alerts: {alert_count}/20")
    print("\nDemonstration complete!")


if __name__ == "__main__":
    main()
