import numpy as np
import torch
import torch.nn as nn
from typing import Dict
from pathlib import Path


class TimesNetEmbedder:

    def __init__(
        self,
        pretrained_path: str = "data/external/pretrained_models/timesnet_pretrained.pth",
        embedding_dim: int = 512,
        input_window_size: int = 96,
        freeze_backbone: bool = True,
    ):
        self.input_window_size = input_window_size
        self.embedding_dim = embedding_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._load_model(pretrained_path)
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.eval()
        self.model.to(self.device)

        print(f"TimesNet загружена на {self.device}, embedding_dim={embedding_dim}")

    def _load_model(self, pretrained_path: str) -> nn.Module:
        path = Path(pretrained_path)
        if path.exists():
            model = torch.load(path, map_location=self.device)
            return model
        else:
            print(f"[WARN] Предобученная модель не найдена: {pretrained_path}")
            print(f"[WARN] Использую упрощенный CNN-экстрактор для демонстрации")
            return self._build_dummy_extractor()

    def _build_dummy_extractor(self) -> nn.Module:
        class DummyTimeSeriesEncoder(nn.Module):
            def __init__(self, input_dim: int, embedding_dim: int):
                super().__init__()
                self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=7, padding=3)
                self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
                self.conv3 = nn.Conv1d(128, embedding_dim, kernel_size=3, padding=1)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.conv3(x)
                x = self.pool(x)
                return x.squeeze(-1)

        return DummyTimeSeriesEncoder(
            input_dim=4,
            embedding_dim=self.embedding_dim,
        )

    def extract_embeddings(
        self, window_data: np.ndarray, prefix: str = "emb_"
    ) -> Dict[str, float]:
        if window_data.shape[0] < self.input_window_size:
            pad_size = self.input_window_size - window_data.shape[0]
            window_data = np.pad(window_data, ((0, pad_size), (0, 0)), mode="constant")

        if window_data.shape[0] > self.input_window_size:
            window_data = window_data[-self.input_window_size:]

        x = torch.FloatTensor(window_data.T).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(x).cpu().numpy().flatten()

        features = {}
        for i, val in enumerate(embedding):
            features[f"{prefix}{i}"] = float(val)

        return features
