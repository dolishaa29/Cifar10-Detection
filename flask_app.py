# flask_app.py
from flask import Flask, request, jsonify
import torch
import io
from utils import preprocess_image_bytes, CIFAR10_CLASSES
from model import CIFARResNet18
import os

app = Flask(__name__)

# Config
MODEL_PATH = os.environ.get("MODEL_PATH", "saved_models/best_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
def load_model(path=MODEL_PATH, device=DEVICE):
    model = CIFARResNet18(num_classes=10)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt else ckpt)
    model.to(device).eval()
    return model

model = load_model()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok"})

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error":"no file uploaded"}), 400
    file = request.files['file']
    img_bytes = file.read()
    try:
        tensor = preprocess_image_bytes(img_bytes)  # 1x3x32x32
    except Exception as e:
        return jsonify({"error":"invalid image", "detail": str(e)}), 400
    tensor = tensor.to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy().tolist()[0]
        pred_idx = int(torch.argmax(outputs, dim=1).cpu().item())
    return jsonify({
        "predicted_class": CIFAR10_CLASSES[pred_idx],
        "predicted_index": pred_idx,
        "probabilities": {CIFAR10_CLASSES[i]: float(probs[i]) for i in range(len(probs))}
    })

if __name__ == "__main__":
    # Example: set host=0.0.0.0 to expose on LAN
    app.run(host="0.0.0.0", port=5000, debug=False)
