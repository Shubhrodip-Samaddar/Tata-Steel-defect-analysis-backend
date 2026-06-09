import torch
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
import timm
import os
import json

CONFIDENCE_THRESHOLD = 0.35

def load_class_names():
    class_file = os.path.join("model", "class_names.json")
    if os.path.exists(class_file):
        with open(class_file, "r") as f:
            return json.load(f)
    return ["crazing", "crease", "crescent_gap", "inclusion", "oil_spot",
            "patches", "pitted_surface", "punching", "rolled-in_scale",
            "scratches", "water_spot", "weld_line"]

CLASS_NAMES = load_class_names()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_model():
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=len(CLASS_NAMES))
    model_path = os.path.join("model", "defect_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"✅ Trained model loaded. Classes: {CLASS_NAMES}")
    else:
        print("⚠️  No trained model found. Using demo mode.")
    model.eval()
    return model

model = load_model()

def quick_preprocess(image: Image.Image) -> Image.Image:
    # Light grayscale conversion
    gray = image.convert("L").convert("RGB")
    # Mild contrast boost only
    gray = ImageEnhance.Contrast(gray).enhance(1.5)
    return gray

def predict_defect(image: Image.Image):
    model_path = os.path.join("model", "defect_model.pth")
    if not os.path.exists(model_path):
        import random
        defect = random.choice(CLASS_NAMES)
        confidence = round(random.uniform(0.80, 0.99), 2)
        return defect, confidence, {c: round(random.uniform(0, 1), 2) for c in CLASS_NAMES}

    # Resize large images
    if max(image.size) > 512:
        image.thumbnail((512, 512), Image.BILINEAR)

    # Run both color and grayscale — pick best
    img_color = image.convert("RGB")
    img_gray  = quick_preprocess(image)

    t_color = transform(img_color).unsqueeze(0)
    t_gray  = transform(img_gray).unsqueeze(0)

    with torch.no_grad():
        out_color = model(t_color)
        out_gray  = model(t_gray)

        prob_color = torch.softmax(out_color, dim=1)[0]
        prob_gray  = torch.softmax(out_gray,  dim=1)[0]

        conf_color, pred_color = torch.max(prob_color, 0)
        conf_gray,  pred_gray  = torch.max(prob_gray,  0)

        if conf_color >= conf_gray:
            confidence    = conf_color
            predicted     = pred_color
            probabilities = prob_color
        else:
            confidence    = conf_gray
            predicted     = pred_gray
            probabilities = prob_gray

    defect    = CLASS_NAMES[predicted.item()]
    all_probs = {CLASS_NAMES[i]: round(probabilities[i].item(), 2) for i in range(len(CLASS_NAMES))}

    if confidence.item() < CONFIDENCE_THRESHOLD:
        return "no_defect", confidence.item(), all_probs

    return defect, confidence.item(), all_probs