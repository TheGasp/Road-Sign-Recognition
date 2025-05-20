import os
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
from PIL import Image
import cv2
import numpy as np

# Ajout du dossier YOLOv5 au PYTHONPATH
sys.path.append("yolov5")

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
RESULT_FOLDER = './static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Chargement du modèle YOLOv5
model_path = 'yolov5/weights/best.pt'
device = select_device('')  # auto GPU/CPU
model = DetectMultiBackend(model_path, device=device)
model.eval()

@app.route('/')
def index():
    return render_template('recognize.html')

@app.route('/recognize')
def recognize():
    return render_template('recognize.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = file.filename
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((640, 640))
    img_array = np.array(img_resized)
    img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1).div(255.0).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        pred = model(img_tensor, augment=False)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # Convert back to original image for annotation
    img_annotated = np.array(img_resized).copy()

    # Draw boxes and save
    if pred[0] is not None and len(pred[0]) > 0:
        for *xyxy, conf, cls in pred[0]:
            label = f"{model.names[int(cls)]} {conf:.2f}"
            xyxy = [int(x.item()) for x in xyxy]
            cv2.rectangle(img_annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(img_annotated, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        result_filename = f"result_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR))

        return jsonify({
            'class': model.names[int(pred[0][0][-1].item())],
            'confidence': float(pred[0][0][4].item()),
            'image_url': f"/static/results/{result_filename}"
        })
    else:
        return jsonify({'error': 'No sign detected'}), 200

if __name__ == '__main__':
    app.run(debug=True)
