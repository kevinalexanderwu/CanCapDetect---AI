"""
CanCap Detect – Python Detection API
Flask API yang menerima gambar dari Laravel dan menjalankan inferensi YOLOv8/YOLOv11
"""

import os
import time
import base64
import io
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# ── Konfigurasi ────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATHS = {
    "yolov8":  MODELS_DIR / "yolov8.pt",
    "yolov11": MODELS_DIR / "yolov11.pt",
}

# Cache model agar tidak reload setiap request
_model_cache: dict[str, YOLO] = {}

# Confidence threshold
CONF_THRESHOLD = 0.4
LABEL_MAP = {
    "Can_Critical_Defect": "Can\nCritical Defect",
    "Can_Major_Defect":    "Can\nMajor Defect",
    "Can_Minor_Defect":    "Can\nMinor Defect",
    "Can_No_Defect":       "Can\nNo Defect",
    "Cap_Defect":          "Cap\nDefect",
    "Cap_Good":            "Cap\nNo Defect",
}


# ── Helper ──────────────────────────────────────────────────────────────────────

def load_model(model_name: str) -> YOLO:
    """Load model YOLO, cache setelah pertama kali dimuat."""
    if model_name not in _model_cache:
        model_path = MODEL_PATHS.get(model_name)
        if not model_path or not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        _model_cache[model_name] = YOLO(str(model_path))
        print(f"[INFO] Model {model_name} loaded from {model_path}")
    return _model_cache[model_name]


def image_to_base64(img_bgr: np.ndarray) -> str:
    """Konversi OpenCV image (BGR) ke base64 JPEG string."""
    _, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode("utf-8")


def run_detection(image_bytes: bytes, model_name: str) -> dict:
    """Jalankan inferensi YOLO dan kembalikan hasil."""
    # Decode image
    nparr  = np.frombuffer(image_bytes, np.uint8)
    img    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Cannot decode image")
    # Resize gambar agar konsisten
    # Resize gambar agar konsisten
    target_width = 640
    if img.shape[1] != target_width:
        scale = target_width / img.shape[1]
        img = cv2.resize(img, (target_width, int(img.shape[0] * scale)))

    # Load model
    model = load_model(model_name)

    # Inferensi
    start_time = time.time()
    results    = model(img, conf=CONF_THRESHOLD, verbose=False)
    elapsed    = round(time.time() - start_time, 3)

    # Parse hasil
    detections  = []
    result_img  = img.copy()

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf  = round(float(box.conf[0]), 3)
            cls   = int(box.cls[0])
            label = LABEL_MAP.get(model.names[cls], model.names[cls]) if model.names else str(cls)

            detections.append({
                "label":      label,
                "confidence": conf,
                "bbox":       [x1, y1, x2, y2],
                "class_id":   cls,
            })

            # Gambar bounding box
            color = (32, 153, 32) if "no defect" in label.lower() else (53, 53, 229)
            text = f"{label} {conf:.0%}"

            # Gambar bounding box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 3)

            # Pakai Pillow untuk font custom
            from PIL import ImageFont, ImageDraw, Image
            font_path = str(BASE_DIR / "fonts" / "DMSans-Bold.ttf")
            font = ImageFont.truetype(font_path, 28)

            img_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            bbox_text = draw.textbbox((0, 0), text, font=font)
            tw = bbox_text[2] - bbox_text[0]
            th = bbox_text[3] - bbox_text[1]

            # Background label
            draw.rectangle([(x1, y1 - th - 14), (x1 + tw + 10, y1)], fill=color[::-1])
            draw.text((x1 + 5, y1 - th - 10), text, font=font, fill=(255, 255, 255))

            result_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    # Hitung total defect (label yang bukan "good")
    defect_labels = [d for d in detections if "no defect" not in d["label"].lower()]

    return {
        "detections":           detections,
        "total_defects":        len(defect_labels),
        "processing_time":      elapsed,
        "result_image_base64":  image_to_base64(result_img),
    }


# ── Routes ──────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_available": list(MODEL_PATHS.keys())})


@app.route("/models", methods=["GET"])
def get_models():
    available = [name for name, path in MODEL_PATHS.items() if path.exists()]
    return jsonify({"models": available})


@app.route("/detect", methods=["POST"])
def detect():
    # Validasi input
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    model_name = request.form.get("model", "yolov8").lower()
    if model_name not in MODEL_PATHS:
        return jsonify({"error": f"Invalid model. Choose from: {list(MODEL_PATHS.keys())}"}), 400

    image_file  = request.files["image"]
    image_bytes = image_file.read()

    try:
        result = run_detection(image_bytes, model_name)
        return jsonify({
            "success":              True,
            "model_used":           model_name,
            "detections":           result["detections"],
            "total_defects":        result["total_defects"],
            "processing_time":      result["processing_time"],
            "result_image_base64":  result["result_image_base64"],
        })

    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 404

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    except Exception as e:
        app.logger.error(f"Detection error: {e}", exc_info=True)
        return jsonify({"success": False, "error": "Internal server error"}), 500


# ── Entry Point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[INFO] CanCap Detect API starting...")
    for name, path in MODEL_PATHS.items():
        status = "✓ found" if path.exists() else "✗ NOT FOUND"
        print(f"  [{status}] {name}: {path}")
    app.run(host="0.0.0.0", port=8001, debug=False)
