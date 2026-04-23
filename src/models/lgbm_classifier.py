import lightgbm as lgb
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from .cost_sensitive import CostSensitiveMetric


class LGBMClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, config_path: str = "configs/model_config.yaml"):
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.lgbm_params = config["lgbm"]
        self.cost_config = config["cost_matrix"]
        self.model = None
        self.best_threshold_ = 0.3

    def fit(self, X, y, eval_set=None):
        train_data = lgb.Dataset(
            X, label=y,
            params={"verbose": -1}
        )
        valid_sets = [train_data]

        if eval_set is not None:
            X_val, y_val = eval_set
            valid_data = lgb.Dataset(
                X_val, label=y_val,
                reference=train_data,
                params={"verbose": -1}
            )
            valid_sets = [train_data, valid_data]

        self.model = lgb.train(
            params=self.lgbm_params,
            train_set=train_data,
            num_boost_round=self.lgbm_params.get("n_estimators", 1000),
            valid_sets=valid_sets,
            callbacks=[lgb.early_stopping(stopping_rounds=self.lgbm_params.get("early_stopping_rounds", 100))]
        )
        return self

    def predict(self, X):
        probs = self.predict_proba(X)
        binary_pred = (probs[:, 1] >= self.best_threshold_).astype(int)
        return binary_pred

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите fit().")
        
        preds = self.model.predict(X)
        if preds.ndim == 1:
            probs = np.column_stack((1 - preds, preds))
        else:
            probs = preds
        return probs
	valid_sets=valid_sets,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=self.lgbm_params.get("early_stopping_rounds", 100)
                ),
                lgb.log_evaluation(period=100),
            ],
        )

        if eval_set is not None:
            self._tune_threshold(X_val, y_val)

        return self

    def _tune_threshold(self, X_val, y_val):
        cost_metric = CostSensitiveMetric(
            false_positive_cost=self.cost_config["false_positive_cost"],
            false_negative_cost=self.cost_config["false_negative_cost"],
            true_positive_gain=self.cost_config["true_positive_gain"],
        )

        y_proba = self.model.predict(X_val)
        if len(y_proba.shape) > 1:
            y_proba = 1 - y_proba[:, 0]

        thresholds = np.arange(0.05, 0.95, 0.05)
        best_cost = -np.inf
        best_threshold = 0.3

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            cost = cost_metric.calculate_cost(y_val.values, y_pred)
            if cost > best_cost:
                best_cost = cost
                best_threshold = threshold

        self.best_threshold_ = best_threshold
        print(f"Оптимальный порог: {best_threshold:.2f}, ожидаемая экономия: ${best_cost:,.0f}")

    def predict(self, X):
        y_proba = self.model.predict(X)
        if len(y_proba.shape) > 1:
            y_proba_risk = 1 - y_proba[:, 0]
            predictions = np.where(y_proba_risk >= self.best_threshold_, 
                                   np.argmax(y_proba[:, 1:], axis=1) + 1, 
                                   0)
        else:
            predictions = (y_proba >= self.best_threshold_).astype(int)
        return predictions

    def predict_proba(self, X):
        return self.model.predict(X)

    def save(self, path: str):
        import joblib
        joblib.dump({
            "model": self.model,
            "best_threshold": self.best_threshold_,
        }, path)

    @classmethod
    def load(cls, path: str):
        import joblib
        data = joblib.load(path)
        instance = cls()
        instance.model = data["model"]
        instance.best_threshold_ = data["best_threshold"]
        return instance
