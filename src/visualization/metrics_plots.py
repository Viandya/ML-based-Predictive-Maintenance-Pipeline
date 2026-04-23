import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
)


class MetricsPlotter:

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

    @staticmethod
    def plot_roc_curve(y_true, y_score, save_path=None):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random model")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

    @staticmethod
    def plot_precision_recall(y_true, y_score, save_path=None):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color="green", lw=2, label=f"PR (AUC = {pr_auc:.3f})")
        ax.axhline(
            y=np.mean(y_true),
            color="red",
            linestyle="--",
            label=f"Baseline (failure rate = {np.mean(y_true):.3f})",
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

    @staticmethod
    def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
        importance = model.model.feature_importance(importance_type="gain")
        indices = np.argsort(importance)[::-1][:top_n]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(top_n), importance[indices][::-1], color="steelblue")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in indices][::-1])
        ax.set_xlabel("Gain")
        ax.set_title(f"Top-{top_n} Feature Importance")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()
