import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from src.models.lgbm_classifier import LGBMClassifier
from src.models.cost_sensitive import CostSensitiveMetric


class TestLGBMClassifier:

    @pytest.fixture
    def sample_data(self):
        X, y = make_classification(
            n_samples=500,
            n_features=20,
            n_classes=3,
            n_informative=10,
            random_state=42,
        )
        return pd.DataFrame(X), pd.Series(y)

    def test_fit_predict(self, sample_data):
        X, y = sample_data
        model = LGBMClassifier()
        model.lgbm_params["n_estimators"] = 50
        model.lgbm_params["num_class"] = 3
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert set(predictions).issubset({0, 1, 2})

    def test_save_load(self, sample_data, tmp_path):
        X, y = sample_data
        model = LGBMClassifier()
        model.lgbm_params["n_estimators"] = 50
        model.lgbm_params["num_class"] = 3
        model.fit(X, y)

        path = tmp_path / "model.joblib"
        model.save(str(path))

        loaded = LGBMClassifier.load(str(path))
        pred_original = model.predict(X)
        pred_loaded = loaded.predict(X)

        np.testing.assert_array_equal(pred_original, pred_loaded)


class TestCostSensitiveMetric:

    def test_perfect_predictions(self):
        metric = CostSensitiveMetric(
            false_positive_cost=500,
            false_negative_cost=10000,
            true_positive_gain=8000,
        )
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        cost = metric.calculate_cost(y_true, y_pred)
        assert cost == 16000

    def test_false_positive_penalty(self):
        metric = CostSensitiveMetric(
            false_positive_cost=500,
            false_negative_cost=10000,
            true_positive_gain=8000,
        )
        y_true = np.array([0, 0, 1])
        y_pred = np.array([1, 0, 0])

        cost = metric.calculate_cost(y_true, y_pred)
        assert cost == -10500

    def test_empty_arrays(self):
        metric = CostSensitiveMetric()
        cost = metric.calculate_cost(np.array([]), np.array([]))
        assert cost == 0
