import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from ..features.deep_embeddings import TimesNetEmbedder


class TimesNetFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, pretrained_path: str = "data/external/pretrained_models/timesnet_pretrained.pth"):
        self.pretrained_path = pretrained_path
        self.embedder = None

    def fit(self, X, y=None):
        self.embedder = TimesNetEmbedder(pretrained_path=self.pretrained_path)
        return self

    def transform(self, X):
        if self.embedder is None:
            raise ValueError("Экстрактор не обучен. Сначала вызовите fit().")

        embeddings_list = []
        for i in range(len(X)):
            window = X.iloc[i:i+1].values
            emb = self.embedder.extract_embeddings(window)
            embeddings_list.append(emb)

        import pandas as pd
        return pd.DataFrame(embeddings_list)
