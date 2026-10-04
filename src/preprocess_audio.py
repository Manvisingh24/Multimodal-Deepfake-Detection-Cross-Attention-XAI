import os
import torch
import librosa
import numpy as np
from pathlib import Path

class AudioPreprocessor:
    def __init__(self, sample_rate=16000, n_mels=128, duration=3.0):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.target_length = int(sample_rate * duration)

    def extract_mel_spectrogram(self, audio_path: str):
        """Loads audio file, normalizes duration, and generates 2D Mel-Spectrogram tensor."""
        try:
            # Load audio waveform at target sample rate
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

            # Pad or truncate waveform to standard length
            if len(y) < self.target_length:
                y = np.pad(y, (0, self.target_length - len(y)), mode='constant')
            else:
                y = y[:self.target_length]

            # Compute Mel-Spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=self.sample_rate, n_mels=self.n_mels, fmax=8000
            )
            
            # Convert power spectrogram to decibels (log-scale)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Min-Max Normalization to [0, 1] range
            normalized_spec = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

            return normalized_spec

        except Exception as e:
            print(f"❌ Audio Preprocessing Error on {audio_path}: {str(e)}")
            return None

if __name__ == "__main__":
    audio_preprocessor = AudioPreprocessor()
    print("✅ AudioPreprocessor initialized successfully with Librosa.")