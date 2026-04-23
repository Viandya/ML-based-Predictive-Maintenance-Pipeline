import mlflow
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Tuple, Optional
from ..models.lgbm_classifier import LGBMClassifier
from ..models.cost_sensitive import CostSensitiveMetric
from .validator import TimeSeriesValidator


class ModelTrainer:

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        model_config_path: str = "configs/model_config.yaml",
    ):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        with open(model_config_path, "r") as f:
            self.model_config = yaml.safe_load(f)

        self.training_config = self.config["training"]
        self.random_state = self.training_config["random_state"]
        self.model_dir = Path(self.config["paths"]["models"])
        self.model_dir.mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(self.config.get("mlflow_uri", "./experiments/mlruns"))
        mlflow.set_experiment("predictive_maintenance")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> LGBMClassifier:
        with mlflow.start_run(run_name="lgbm_baseline"):
            mlflow.log_params(self.model_config["lgbm"])
            mlflow.log_params(self.model_config["cost_matrix"])

            model = LGBMClassifier()

            eval_set = None
            if X_val is not None and y_val is not None:
                eval_set = (X_val, y_val)

            model.fit(X_train, y_train, eval_set=eval_set)

            if X_val is not None and y_val is not None:
                y_pred = model.predict(X_val)
                validator = TimeSeriesValidator()
                metrics = validator.get_metrics(y_val.values, y_pred)

                cost_metric = CostSensitiveMetric(
                    false_positive_cost=self.model_config["cost_matrix"]["false_positive_cost"],
                    false_negative_cost=self.model_config["cost_matrix"]["false_negative_cost"],
                    true_positive_gain=self.model_config["cost_matrix"]["true_positive_gain"],
                )
                total_cost = cost_metric.calculate_cost(y_val.values, y_pred)

                mlflow.log_metrics(metrics)
                mlflow.log_metric("total_cost_savings", total_cost)

                print("\n" + "=" * 50)
                print("РЕЗУЛЬТАТЫ НА ВАЛИДАЦИИ:")
                print(f"  Recall (обнаружение отказов):    {metrics['recall']:.3f}")
                print(f"  Precision (точность тревог):     {metrics['precision']:.3f}")
                print(f"  False Positive Rate:             {metrics['false_positive_rate']:.3f}")
                print(f"  Ложных тревог:                   {metrics['false_positives']}")
                print(f"  Пропущено отказов:               {metrics['false_negatives']}")
                print(f"  Ожидаемая экономия:              ${total_cost:,.0f}")
                print("=" * 50)

            model_path = self.model_dir / "lgbm_model.joblib"
            model.save(str(model_path))
            mlflow.log_artifact(str(model_path))

            return model

    def cross_validate(self, df: pd.DataFrame, feature_cols: list, target_col: str) -> dict:
        validator = TimeSeriesValidator(n_splits=self.training_config["cv_folds"])
        cv_scores = []

        for fold_idx, (train_df, val_df) in enumerate(
            validator.split(df, date_col="datetime")
        ):
            print(f"\nFold {fold_idx + 1}/{self.training_config['cv_folds']}")
            print(f"  Train: {train_df['datetime'].min()} -> {train_df['datetime'].max()}")
            print(f"  Val:   {val_df['datetime'].min()} -> {val_df['datetime'].max()}")

            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_val = val_df[feature_cols]
            y_val = val_df[target_col]

            model = LGBMClassifier()
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
            y_pred = model.predict(X_val)

            metrics = validator.get_metrics(y_val.values, y_pred)
            cv_scores.append(metrics)

        avg_metrics = {}
        for key in cv_scores[0].keys():
            values = [fold[key] for fold in cv_scores]
            avg_metrics[f"mean_{key}"] = np.mean(values)
            avg_metrics[f"std_{key}"] = np.std(values)

        print("\n" + "=" * 50)
        print("КРОСС-ВАЛИДАЦИЯ:")
        print(f"  Средний Recall:    {avg_metrics['mean_recall']:.3f} ± {avg_metrics['std_recall']:.3f}")
        print(f"  Средний Precision: {avg_metrics['mean_precision']:.3f} ± {avg_metrics['std_precision']:.3f}")
        print(f"  Средний F1:        {avg_metrics['mean_f1']:.3f} ± {avg_metrics['std_f1']:.3f}")
        print("=" * 50)

        return avg_metrics
