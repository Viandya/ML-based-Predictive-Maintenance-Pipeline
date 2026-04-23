import numpy as np
from sklearn.metrics import make_scorer


class CostSensitiveMetric:

    def __init__(
        self,
        false_positive_cost: float = 500,
        false_negative_cost: float = 10000,
        true_positive_gain: float = 8000,
    ):
        self.fp_cost = false_positive_cost
        self.fn_cost = false_negative_cost
        self.tp_gain = true_positive_gain

    def calculate_cost(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        y_true_binary = (y_true != 0).astype(int)
        y_pred_binary = (y_pred != 0).astype(int)

        tp = ((y_true_binary == 1) & (y_pred_binary == 1)).sum()
        fp = ((y_true_binary == 0) & (y_pred_binary == 1)).sum()
        fn = ((y_true_binary == 1) & (y_pred_binary == 0)).sum()

        total_cost = (tp * self.tp_gain) - (fp * self.fp_cost) - (fn * self.fn_cost)
        return total_cost

    def make_scorer(self):
        def cost_score(estimator, X, y):
            y_pred = estimator.predict(X)
            return self.calculate_cost(y.values, y_pred)

        return make_scorer(cost_score, greater_is_better=True)
