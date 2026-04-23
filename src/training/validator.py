import numpy as np
import pandas as pd
from typing import Iterator, Tuple
from sklearn.model_selection import TimeSeriesSplit


class TimeSeriesValidator:

    def __init__(self, n_splits: int = 5, gap: int = 0):
        self.n_splits = n_splits
        self.gap = gap
        self.tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    def split(
        self, df: pd.DataFrame, date_col: str = "datetime"
    ) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        df_sorted = df.sort_values(date_col).reset_index(drop=True)

        for train_idx, val_idx in self.tscv.split(df_sorted):
            train_df = df_sorted.iloc[train_idx]
            val_df = df_sorted.iloc[val_idx]

            max_train_date = train_df[date_col].max()
            val_df = val_df[val_df[date_col] > max_train_date]

            if len(val_df) == 0:
                continue

            yield train_df, val_df

    def get_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
        )

        y_true_bin = (y_true != 0).astype(int)
        y_pred_bin = (y_pred != 0).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()

        return {
            "accuracy": accuracy_score(y_true_bin, y_pred_bin),
            "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
            "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
            "f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }
