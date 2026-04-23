import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from typing import Dict


class SpectralFeatureExtractor:

    def __init__(self, n_components: int = 10, sampling_rate: float = 1.0):
        self.n_components = n_components
        self.sampling_rate = sampling_rate

    def extract_fft_features(self, window_data: np.ndarray, prefix: str = "") -> Dict[str, float]:
        features = {}

        if len(window_data) < 10:
            for i in range(self.n_components):
                features[f"{prefix}fft_freq_{i}"] = 0.0
                features[f"{prefix}fft_amp_{i}"] = 0.0
            return features

        n = len(window_data)
        fft_values = fft(window_data)
        freqs = fftfreq(n, d=1/self.sampling_rate)

        positive_mask = freqs > 0
        positive_freqs = freqs[positive_mask]
        positive_amps = np.abs(fft_values[positive_mask])

        if len(positive_amps) >= self.n_components:
            top_indices = np.argsort(positive_amps)[-self.n_components:][::-1]
            for i, idx in enumerate(top_indices):
                features[f"{prefix}fft_freq_{i}"] = float(positive_freqs[idx])
                features[f"{prefix}fft_amp_{i}"] = float(positive_amps[idx])
        else:
            for i in range(self.n_components):
                if i < len(positive_amps):
                    features[f"{prefix}fft_freq_{i}"] = float(positive_freqs[i])
                    features[f"{prefix}fft_amp_{i}"] = float(positive_amps[i])
                else:
                    features[f"{prefix}fft_freq_{i}"] = 0.0
                    features[f"{prefix}fft_amp_{i}"] = 0.0

        return features

    def extract_psd_features(self, window_data: np.ndarray, prefix: str = "") -> Dict[str, float]:
        features = {}

        if len(window_data) < 10:
            features[f"{prefix}psd_max"] = 0.0
            features[f"{prefix}psd_mean"] = 0.0
            features[f"{prefix}psd_total"] = 0.0
            return features

        freqs, psd = welch(window_data, fs=self.sampling_rate, nperseg=min(256, len(window_data)))

        features[f"{prefix}psd_max"] = float(np.max(psd))
        features[f"{prefix}psd_mean"] = float(np.mean(psd))
        features[f"{prefix}psd_total"] = float(np.sum(psd))

        return features

    def extract_all(self, window_data: np.ndarray, prefix: str = "") -> Dict[str, float]:
        features = {}
        features.update(self.extract_fft_features(window_data, prefix))
        features.update(self.extract_psd_features(window_data, prefix))
        return features
