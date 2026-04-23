"""
Main model training script.
Run: python scripts/train_pipeline.py
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.augmenter import DataAugmenter
from src.features.rolling_aggregator import RollingAggregator
from src.features.deep_embeddings import TimesNetEmbedder
from src.training.trainer import ModelTrainer
from src.training.validator import TimeSeriesValidator
import pandas as pd
import yaml


def main():
    parser = argparse.ArgumentParser(description="Train predictive maintenance model")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config")
    parser.add_argument("--skip-augmentation", action="store_true", help="Skip augmentation")
    parser.add_argument("--skip-timesnet", action="store_true", help="Skip TimesNet embeddings")
    parser.add_argument("--cross-validate", action="store_true", help="Run cross-validation")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("TRAINING PIPELINE: PREDICTIVE MAINTENANCE")
    print("=" * 60)

    print("\n[1/6] Loading data...")
    loader = DataLoader(args.config)
    telemetry, errors, machines = loader.load_all()

    print("\n[2/6] Data preprocessing...")
    preprocessor = DataPreprocessor(args.config)
    telemetry = preprocessor.clean_telemetry(telemetry)
    telemetry = preprocessor.merge_errors(telemetry, errors)

    if not args.skip_augmentation:
        print("\n[3/6] Data augmentation...")
        augmenter = DataAugmenter(noise_level=0.02, random_state=42)
        telemetry = augmenter.add_sensor_noise(telemetry)
        telemetry = augmenter.add_random_walk_drift(telemetry, drift_std=0.001)
        telemetry = augmenter.undersample_healthy(telemetry, target_ratio=0.001)

    print("\n[4/6] Feature engineering...")
    sensor_cols = config["data"].get("sensor_cols", ["volt", "rotate", "pressure", "vibration"])

    aggregator = RollingAggregator()
    feature_dfs = []

    for machine_id in telemetry["machineID"].unique():
        machine_df = telemetry[telemetry["machineID"] == machine_id].sort_values("datetime")
        features = aggregator.aggregate_machine(machine_df, sensor_cols)
        feature_dfs.append(features)

        if machine_id % 10 == 0:
            print(f"  Processed machines: {machine_id}")

    features_df = pd.concat(feature_dfs, ignore_index=True)
    print(f"  Total features generated: {features_df.shape[1] - 3}")

    if not args.skip_timesnet:
        print("\n  Extracting TimesNet embeddings...")
        timesnet_config = config.get("timesnet", {})
        if timesnet_config.get("enabled", True):
            embedder = TimesNetEmbedder(
                pretrained_path=timesnet_config.get("pretrained_path", ""),
                embedding_dim=timesnet_config.get("embedding_dim", 512),
                input_window_size=timesnet_config.get("input_window_size", 96),
            )

            embeddings_list = []
            for machine_id in telemetry["machineID"].unique():
                machine_df = telemetry[telemetry["machineID"] == machine_id][sensor_cols].values
                
                window_size = timesnet_config.get("input_window_size", 96)
                for i in range(len(machine_df)):
                    start = max(0, i - window_size + 1)
                    window = machine_df[start:i+1]
                    emb = embedder.extract_embeddings(window)
                    emb["machineID"] = machine_id
                    emb["datetime"] = telemetry[telemetry["machineID"] == machine_id].iloc[i]["datetime"]
                    embeddings_list.append(emb)

                if machine_id % 10 == 0:
                    print(f"    Embeddings for machine {machine_id}")

            embeddings_df = pd.DataFrame(embeddings_list)
            features_df = features_df.merge(
                embeddings_df,
                on=["machineID", "datetime"],
                how="left",
            )
            print(f"  Embeddings added: {embeddings_df.shape[1] - 2}")

    processed_path = Path(config["paths"]["processed_data"])
    processed_path.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(processed_path / "features.parquet")
    print(f"  Features saved: {processed_path / 'features.parquet'}")

    print("\n[5/6] Train/val/test split...")
    features_df = features_df.sort_values("datetime")

    train_ratio = config["split"]["train_ratio"]
    val_ratio = config["split"]["val_ratio"]
    
    n = len(features_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = features_df.iloc[:train_end]
    val_df = features_df.iloc[train_end:val_end]
    test_df = features_df.iloc[val_end:]

    print(f"  Train: {len(train_df):,} ({train_df['datetime'].min()} -> {train_df['datetime'].max()})")
    print(f"  Val:   {len(val_df):,} ({val_df['datetime'].min()} -> {val_df['datetime'].max()})")
    print(f"  Test:  {len(test_df):,} ({test_df['datetime'].min()} -> {test_df['datetime'].max()})")

    train_df.to_parquet(processed_path / "train.parquet")
    val_df.to_parquet(processed_path / "val.parquet")
    test_df.to_parquet(processed_path / "test.parquet")

    drop_cols = ["datetime", "machineID", "failure_component"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    target_col = "failure_component"

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    print("\n[6/6] Training model...")
    trainer = ModelTrainer(args.config)

    if args.cross_validate:
        cv_metrics = trainer.cross_validate(train_df, feature_cols, target_col)

    model = trainer.train(X_train, y_train, X_val, y_val)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Model saved to: {config['paths']['models']}/lgbm_model.joblib")
    print("=" * 60)


if __name__ == "__main__":
    main()
