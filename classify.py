import torch
from PIL import Image
from torchvision import transforms
from model import get_model
from config import IMAGE_SIZE
import sys
import os
import warnings

# Suppress specific torchvision warnings about deprecated 'pretrained'
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


class_names = ['non_perforated', 'perforated']

model = get_model()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_image(image_path):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def predict(image_path):
    image = load_image(image_path)
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1).squeeze()
        pred = output.argmax(1).item()
        confidence = probabilities[pred].item()

    return class_names[pred], confidence, probabilities

if __name__ == "__main__":
    print("✅ Script started...")

    if len(sys.argv) != 2:
        print("Usage: python3 classify.py path/to/image")
        sys.exit(1)

    img_path = sys.argv[1]
    print(f"📷 Image path received: {img_path}")

    if not os.path.exists(img_path):
        print(f" Error: {img_path} does not exist.")
        sys.exit(1)

    label, confidence, probabilities = predict(img_path)

    print(f"\Predicted class: {label} ({confidence * 100:.2f}%)")
    print(" Class probabilities:")
    for i, prob in enumerate(probabilities):
        print(f" - {class_names[i]}: {prob.item() * 100:.2f}%")

