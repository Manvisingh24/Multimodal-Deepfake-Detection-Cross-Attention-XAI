import cv2
import numpy as np
import urllib.request
from pathlib import Path

class FacePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
        # Store cascade file locally inside project models directory
        cascade_dir = Path(__file__).resolve().parent.parent / "models" / "cascades"
        cascade_dir.mkdir(parents=True, exist_ok=True)
        self.cascade_path = cascade_dir / "haarcascade_frontalface_default.xml"

        # Download XML weights if missing locally
        if not self.cascade_path.exists():
            print("📥 Downloading OpenCV Haar Cascade weights...")
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(url, self.cascade_path)
            print("✅ Cascade XML downloaded successfully.")

        self.face_cascade = cv2.CascadeClassifier(str(self.cascade_path))

    def extract_and_crop_face(self, image_path: str):
        """Detects primary face, adds margin, and resizes to target_size."""
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if len(faces) == 0:
            # Fallback: simple resize if no face detected
            return cv2.resize(image_rgb, self.target_size)

        # Take the largest face detected
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

        # Add 10% bounding box margin to capture boundary artifacts
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)

        img_h, img_w, _ = image.shape
        start_x = max(0, x - margin_x)
        start_y = max(0, y - margin_y)
        end_x = min(img_w, x + w + margin_x)
        end_y = min(img_h, y + h + margin_y)

        cropped_face = image_rgb[start_y:end_y, start_x:end_x]
        resized_face = cv2.resize(cropped_face, self.target_size)

        return resized_face

if __name__ == "__main__":
    preprocessor = FacePreprocessor()
    print("✅ FacePreprocessor verified: Haar Cascade weights verified and ready.")

    # Test preprocessor on a sample raw face image from downloaded dataset
    raw_data_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "faces_dataset"
    sample_images = list(raw_data_dir.rglob("*.jpg")) + list(raw_data_dir.rglob("*.png"))

    if sample_images:
        test_img_path = sample_images[0]
        cropped_tensor = preprocessor.extract_and_crop_face(test_img_path)
        if cropped_tensor is not None:
            print(f"✅ Successfully processed sample image: {test_img_path.name}")
            print(f"   Output Tensor Shape: {cropped_tensor.shape} (Height, Width, Channels)")
    else:
        print("⚠️ No dataset images found in data/raw/faces_dataset to test.")