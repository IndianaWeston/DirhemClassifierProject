import torch
from PIL import Image
from torchvision import transforms
from model import get_model
from config import IMAGE_SIZE
import sys
import os

# Define your class names here
class_names = ['non_perforated', 'perforated']

# Load model
model = get_model()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and prepare image (handles .tiff, .jpg, etc.)
def load_image(image_path):
    image = Image.open(image_path)
    
    # Convert all non-RGB formats (e.g., grayscale, CMYK)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    return image

# Prediction function
def predict(image_path):
    image = load_image(image_path)
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(1).item()
    
    return class_names[pred]

# CLI interface
if __name__ == "__main__":
    print("Script started...")

    if len(sys.argv) != 2:
        print("Usage: python3 classify.py path/to/image.[jpg|tiff]")
        sys.exit(1)

    img_path = sys.argv[1]
    print(f"📷 Image path received: {img_path}")

    if not os.path.exists(img_path):
        print(f"Error: {img_path} does not exist.")
        sys.exit(1)

    result = predict(img_path)
    print("Predicted class:", result)
