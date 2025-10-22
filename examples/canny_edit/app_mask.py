import argparse
import base64
import os
import sys

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

import mindspore as ms

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sam_dir = os.path.join(parent_dir, "sam2")
sys.path.insert(0, sam_dir)

# Check if we can use the model of SAM2.1
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MASK_TEMP_FOLDER"] = "mask_temp"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Create the directory of uploading
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["MASK_TEMP_FOLDER"], exist_ok=True)


def create_ellipse_mask(points, image_shape):
    """create oval mask according to trace of points"""
    if len(points) < 5:
        # If points are much less, create a circle
        center = np.mean(points, axis=0)
        radius = max(10, int(np.max(np.abs(points - center)) / 2))
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.circle(mask, (int(center[0]), int(center[1])), radius, 255, -1)
        return mask

    # fit oval
    points = np.array(points, dtype=np.int32)
    ellipse = cv2.fitEllipse(points)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, ellipse, 255, -1)
    return mask


def create_sam_mask(points, image, min_mask_region_area=500):
    """Create mask using SAM2.1"""
    if not SAM_AVAILABLE or sam_predictor is None:
        raise Exception("SAM2.1 model is not available")

    # Set image
    sam_predictor.set_image(image)

    # Prepare input points
    input_points = np.array(points)
    input_labels = np.ones(len(points))  # All points are front view

    # Predict mask
    masks, scores, _ = sam_predictor.predict(
        point_coords=input_points, point_labels=input_labels, multimask_output=True
    )

    # Choose one mask with the highest score
    best_mask_idx = np.argmax(scores)
    mask = masks[best_mask_idx]
    mask = mask.astype(np.uint8)
    mask = remove_small_regions(mask, min_mask_region_area, "holes")
    mask = remove_small_regions(mask, min_mask_region_area, "islands")

    # Transfer into uint8
    mask = (mask * 255).astype(np.uint8)
    return mask


def remove_small_regions(mask, area_thresh, mode):
    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask


@app.route("/")
def index():
    return render_template("index.html", sam_available=SAM_AVAILABLE)


@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    # Save uploading image
    filename = file.filename
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Read an image and get its size
    image = cv2.imread(filepath)
    if image is None:
        return jsonify({"error": "Invalid image file"}), 400

    height, width = image.shape[:2]

    return jsonify({"filename": filename, "filepath": filepath, "width": width, "height": height})


@app.route("/process", methods=["POST"])
def process_image():
    data = request.json
    filename = data.get("filename")
    points = data.get("points", [])
    method = data.get("method", "ellipse")  # 'ellipse' or 'sam'

    if not filename or not points:
        return jsonify({"error": "Missing filename or points"}), 400

    # Read an image
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image = cv2.imread(filepath)
    if image is None:
        return jsonify({"error": "Image not found"}), 404

    try:
        # Create mask following chosen mask
        if method == "ellipse":
            mask = create_ellipse_mask(points, image.shape)
        elif method == "sam" and SAM_AVAILABLE:
            mask = create_sam_mask(points, image)
        else:
            return jsonify({"error": "Invalid method or SAM not available"}), 400

        # Save mask as a temporary file
        mask_filename = f"mask_{filename}"
        mask_file = mask_filename.split(".")[0]
        count = 0
        for file in os.listdir(app.config["MASK_TEMP_FOLDER"]):
            if mask_file in file:
                count += 1

        mask_path = os.path.join(app.config["MASK_TEMP_FOLDER"], f"{mask_file}_{method}_{count}.png")

        cv2.imwrite(mask_path, mask)

        # Transfer mask into encoding of base64
        _, buffer = cv2.imencode(".png", mask)
        mask_base64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify(
            {
                "mask_filename": mask_filename,
                "mask_path": mask_path,
                "mask_data": f"data: image/png; base64, {mask_base64}",
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download/<filename>")
def download_mask(filename):
    mask_path = os.path.join(app.config["MASK_TEMP_FOLDER"], filename)
    if os.path.exists(mask_path):
        return send_file(mask_path, as_attachment=True)
    return jsonify({"error": "File not found"}), 404


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam_checkpoint", type=str, default="./checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--model_cfg", type=str, default="./configs/sam2.1/sam2.1_hiera_l.yaml")
    args = parser.parse_args()

    # try to load model of SAM2.1
    sam_model = None
    sam_predictor = None
    if SAM_AVAILABLE:
        try:
            # Note: you should download model weights of SAM2.1, and set the right path
            sam_checkpoint = args.sam_checkpoint

            if os.path.exists(sam_checkpoint):
                model_cfg = args.model_cfg
                sam = build_sam2(model_cfg, sam_checkpoint)
                dtype = ms.float16
                sam.to_float(dtype)
                sam_predictor = SAM2ImagePredictor(sam)
                print("SAM2.1 model loaded successfully")
            else:
                print(f"SAM2.1 model checkpoint not found at {sam_checkpoint}")
                SAM_AVAILABLE = False
        except Exception as e:
            print(f"Error loading SAM2.1 model: {e}")
            SAM_AVAILABLE = False
    else:
        print("SAM2.1 is not available. Please install segment-anything package.")
    app.run(host="0.0.0.0", port=5000, debug=True)
