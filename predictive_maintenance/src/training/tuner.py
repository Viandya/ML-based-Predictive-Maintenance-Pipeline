import optuna
import numpy as np
import yaml
from ..models.lgbm_classifier import LGBMClassifier
from ..models.cost_sensitive import CostSensitiveMetric


class HyperparameterTuner:

    def __init__(self, model_config_path: str = "configs/model_config.yaml"):
        with open(model_config_path, "r") as f:
            self.model_config = yaml.safe_load(f)
        self.cost_metric = CostSensitiveMetric(
            false_positive_cost=self.model_config["cost_matrix"]["false_positive_cost"],
            false_negative_cost=self.model_config["cost_matrix"]["false_negative_cost"],
            true_positive_gain=self.model_config["cost_matrix"]["true_positive_gain"],
        )

    def _objective(self, trial, X_train, y_train, X_val, y_val):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        }

        base_params = self.model_config["lgbm"].copy()
        base_params.update(params)

        model = LGBMClassifier()
        model.lgbm_params = base_params
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        y_pred = model.predict(X_val)

        cost = self.cost_metric.calculate_cost(y_val.values, y_pred)
        return cost

    def tune(
        self, X_train, y_train, X_val, y_val, n_trials: int = 50
    ) -> dict:
        study = optuna.create_study(
            direction="maximize",
            study_name="predictive_maintenance_tuning",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )

        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        print(f"\nЛучшие параметры: {study.best_params}")
        print(f"Лучшая бизнес-метрика: ${study.best_value:,.0f}")

        return study.best_params
