import os
import sys
import torch
from pathlib import Path
import numpy as np

# Ensure src directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from models import MultimodalDeepfakeDetector
from preprocess_face import FacePreprocessor
from preprocess_audio import AudioPreprocessor
import torchvision.transforms as T

def predict_sample(video_path: str, audio_path: str):
    print("🔍 Initializing Phase 7: Inference Pipeline...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Using device: {device}")

    # 1. Load Model Architecture
    model = MultimodalDeepfakeDetector().to(device)

    # 2. Load Saved Model Weights from Phase 6
    model_path = Path(__file__).resolve().parent.parent / "models" / "best_deepfake_model.pth"
    if not model_path.exists():
        print(f"❌ Error: Model weights not found at {model_path}. Run train.py first!")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    print(f"✅ Loaded trained weights from: {model_path.name}")

    # 3. Initialize Preprocessors
    face_preprocessor = FacePreprocessor(target_size=(224, 224))
    face_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Process Inputs Safely
    try:
        face_img = face_preprocessor.extract_and_crop_face(video_path)
    except Exception:
        face_img = None

    if face_img is None:
        face_img = np.zeros((224, 224, 3), dtype=np.uint8)

    face_tensor = face_transform(face_img).unsqueeze(0).to(device) # Shape: [1, 3, 224, 224]

    try:
        mel_spec = process_audio(audio_path, target_sr=16000, duration=3.0, n_mels=128)
    except Exception:
        mel_spec = None

    if mel_spec is None:
        mel_spec = np.zeros((128, 94), dtype=np.float32)

    audio_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) # Shape: [1, 1, 128, 94]

    # 5. Run Inference (No gradient calculation needed)
    with torch.no_grad():
        output_logit = model(face_tensor, audio_tensor)
        probability = torch.sigmoid(output_logit).item()

    # 6. Interpret Results
    prediction_label = "FAKE" if probability >= 0.5 else "REAL"
    confidence = probability if prediction_label == "FAKE" else (1.0 - probability)

    print("\n📊 --- INFERENCE RESULT ---")
    print(f"Target Video: {video_path}")
    print(f"Target Audio: {audio_path}")
    print(f"Prediction:   {prediction_label}")
    print(f"Confidence:   {confidence * 100:.2f}% (Raw Probability: {probability:.4f})")
    print("----------------------------\n")

if __name__ == "__main__":
    # Test inference using dummy paths
    predict_sample("test_video.mp4", "test_audio.wav")