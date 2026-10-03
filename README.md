# Multimodal Deepfake Detection via Cross-Attention Fusion and XAI Heatmaps

A real-world deep learning framework for detecting video and audio deepfakes using a dual-stream architecture, Cross-Attention Transformer fusion, and Grad-CAM visual explainability.

## 📌 Architecture Highlights
- **Visual Stream:** EfficientNet-B4 feature extraction on cropped facial regions via MediaPipe.
- **Audio Stream:** ResNet-18 feature extraction on Mel-Spectrograms generated via Librosa.
- **Cross-Modal Fusion:** Multi-Head Cross-Attention correlates audio and visual feature streams to identify temporal mismatches.
- **Explainable AI (XAI):** Grad-CAM heatmaps providing visual proof of manipulations.
- **Backend API:** Production FastAPI REST server.