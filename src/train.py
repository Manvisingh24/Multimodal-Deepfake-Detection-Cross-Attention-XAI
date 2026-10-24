import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

# Ensure src directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from models import MultimodalDeepfakeDetector
from dataset import DeepfakeDataset

def train_model():
    print("🚀 Initializing Phase 6: Training & Validation Pipeline...")

    # 1. Device configuration (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Using device: {device}")

    # 2. Hyperparameters
    BATCH_SIZE = 4
    EPOCHS = 3
    LEARNING_RATE = 1e-4

    # 3. Initialize Model and Move to Device
    model = MultimodalDeepfakeDetector().to(device)

    # 4. Loss Function and Optimizer
    # BCEWithLogitsLoss combines a Sigmoid layer and Binary Cross Entropy in one single class
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    # 5. Set up Dataset and DataLoader 
    # (Using dummy samples for initial pipeline validation; swap with real data paths later)
    train_samples = [
        {'video_path': 'non_existent_1.mp4', 'audio_path': 'non_existent_1.wav', 'label': 0},
        {'video_path': 'non_existent_2.mp4', 'audio_path': 'non_existent_2.wav', 'label': 1},
        {'video_path': 'non_existent_3.mp4', 'audio_path': 'non_existent_3.wav', 'label': 0},
        {'video_path': 'non_existent_4.mp4', 'audio_path': 'non_existent_4.wav', 'label': 1},
    ]

    train_dataset = DeepfakeDataset(train_samples)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 6. Model Checkpoint Directory
    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / "best_deepfake_model.pth"

    # 7. Training Loop
    print("\nStarting Training Epochs...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        for batch_idx, (face_b, audio_b, label_b) in enumerate(train_loader):
            # Move tensors to device
            face_b = face_b.to(device)
            audio_b = audio_b.to(device)
            label_b = label_b.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(face_b, audio_b)

            # Calculate loss
            loss = criterion(outputs, label_b)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item() * face_b.size(0)
            
            # Calculate accuracy
            preds = torch.sigmoid(outputs) >= 0.5
            correct_preds += (preds == label_b).sum().item()
            total_samples += label_b.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_preds / total_samples

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc*100:.2f}%")

    # 8. Save Trained Model Weights
    torch.save(model.state_dict(), best_model_path)
    print(f"\n✅ Training completed successfully! Model weights saved to: {best_model_path}")

if __name__ == "__main__":
    train_model()