
from pathlib import Path
from io import BytesIO
import uuid

import numpy as np
import tensorflow as tf

from flask import Flask, jsonify, request, send_from_directory, url_for
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

# -------------------------------
# Configuration
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "oral_cancer.keras"
GRADCAM_DIR = BASE_DIR / "generated" / "gradcam"

IMAGE_SIZE = (224, 224)
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES

CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]
    }
})

# -------------------------------
# Load model once
# -------------------------------
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}\n"
        "Run python train_model.py first."
    )

model = tf.keras.models.load_model(MODEL_PATH)
print("CNN model loaded successfully!")


def error_response(code, message, status):
    return jsonify({
        "error": {
            "code": code,
            "message": message
        }
    }), status


# -------------------------------
# Grad-CAM initialization
# -------------------------------
# The saved model has this structure:
# Input -> Sequential preprocessing -> MobileNetV2 -> GAP -> Dropout -> Dense
# Grad-CAM must tap a convolutional layer INSIDE MobileNetV2 while
# keeping the original preprocessing and classification head connected.

base_model = model.get_layer("mobilenetv2_1.00_224")
preprocessing_layer = model.get_layer("sequential")


def find_last_conv_layer(network):
    """Find the last convolutional layer, including nested Keras models."""
    for layer in reversed(network.layers):
        if isinstance(layer, tf.keras.Model):
            found = find_last_conv_layer(layer)
            if found is not None:
                return found

        if isinstance(
            layer,
            (
                tf.keras.layers.Conv2D,
                tf.keras.layers.DepthwiseConv2D,
                tf.keras.layers.SeparableConv2D,
            ),
        ):
            return layer

    return None


last_conv_layer = find_last_conv_layer(base_model)

if last_conv_layer is None:
    raise ValueError(
        "No convolutional layer was found inside MobileNetV2."
    )

print("Grad-CAM target layer:", last_conv_layer.name)

# A model exposing both the final convolutional feature maps and
# MobileNetV2's normal output. Both outputs share the same graph.
backbone_gradcam_model = tf.keras.Model(
    inputs=base_model.inputs,
    outputs=[last_conv_layer.output, base_model.output],
    name="mobilenetv2_gradcam_extractor",
)

# Connect the original input through its original preprocessing block,
# the MobileNetV2 feature extractor, and the original classification head.
gradcam_input = model.inputs[0]
x = preprocessing_layer(gradcam_input, training=False)
conv_outputs, x = backbone_gradcam_model(x, training=False)

# Apply the same head layers used by the original model, after MobileNetV2.
base_index = model.layers.index(base_model)
for layer in model.layers[base_index + 1:]:
    x = layer(x, training=False)

gradcam_model = tf.keras.Model(
    inputs=gradcam_input,
    outputs=[conv_outputs, x],
    name="oral_cancer_gradcam_model",
)

print("Grad-CAM model initialized successfully!")


# -------------------------------
# Create and save Grad-CAM image
# -------------------------------
def generate_gradcam(image_array, cancer_score, predicted_class):
    """
    Generate a Grad-CAM overlay from the selected MobileNetV2 layer.

    image_array must be RGB, resized to 224x224, float32 in [0, 255].
    The original model's preprocessing layer is applied by gradcam_model.
    """
    image_tensor = tf.convert_to_tensor(
        np.expand_dims(image_array, axis=0),
        dtype=tf.float32,
    )

    with tf.GradientTape() as tape:
        feature_maps, prediction = gradcam_model(
            image_tensor, training=False
        )

        # Dense sigmoid output is the Cancer score.
        # For Non-Cancer, explain the complementary class score.
        if predicted_class == "Cancer":
            target_score = prediction[:, 0]
        else:
            target_score = 1.0 - prediction[:, 0]

    gradients = tape.gradient(target_score, feature_maps)

    if gradients is None:
        raise ValueError(
            "Gradients could not be calculated for Grad-CAM."
        )

    # Average gradients spatially to obtain channel importance weights.
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    feature_map = feature_maps[0]
    channel_weights = weights[0]

    heatmap = tf.reduce_sum(
        feature_map * channel_weights,
        axis=-1,
    )
    heatmap = tf.maximum(heatmap, 0.0)

    max_value = tf.reduce_max(heatmap)
    heatmap = heatmap / (
        max_value + tf.keras.backend.epsilon()
    )

    heatmap_array = heatmap.numpy()

    heatmap_image = Image.fromarray(
        np.uint8(np.clip(heatmap_array, 0, 1) * 255),
        mode="L",
    ).resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR,
    )

    heat = np.asarray(
        heatmap_image, dtype=np.float32
    ) / 255.0

    # Blue -> cyan -> yellow -> red color map.
    red = np.clip(1.5 - np.abs(4 * heat - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * heat - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * heat - 1), 0, 1)

    colored_heatmap = np.stack(
        [red, green, blue], axis=-1
    )

    original = np.clip(image_array / 255.0, 0, 1)
    overlay = (
        0.55 * original
        + 0.45 * colored_heatmap
    )

    overlay = np.uint8(
        np.clip(overlay, 0, 1) * 255
    )

    filename = f"gradcam_{uuid.uuid4().hex}.jpg"
    output_path = GRADCAM_DIR / filename

    Image.fromarray(overlay, mode="RGB").save(
        output_path,
        format="JPEG",
        quality=92,
    )

    return filename


# -------------------------------
# Health-check endpoint
# -------------------------------
@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "model_version": "v1",
        "gradcam_enabled": True
    })


# -------------------------------
# Serve generated Grad-CAM images
# -------------------------------
@app.get("/api/gradcam/<path:filename>")
def serve_gradcam(filename):
    # Only serve files from the dedicated generated folder.
    return send_from_directory(
        GRADCAM_DIR,
        filename,
        mimetype="image/jpeg",
        max_age=0
    )


# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/api/v1/predictions")
def predict():
    if "image" not in request.files:
        return error_response(
            "INVALID_REQUEST",
            "Upload an image using the field named 'image'.",
            400
        )

    uploaded_file = request.files["image"]

    if not uploaded_file.filename:
        return error_response(
            "INVALID_REQUEST",
            "Please select an image.",
            400
        )

    raw_bytes = uploaded_file.stream.read(
        MAX_IMAGE_BYTES + 1
    )

    if not raw_bytes:
        return error_response(
            "INVALID_REQUEST",
            "The uploaded file is empty.",
            400
        )

    if len(raw_bytes) > MAX_IMAGE_BYTES:
        return error_response(
            "IMAGE_TOO_LARGE",
            "The maximum image size is 10 MB.",
            413
        )

    try:
        with Image.open(BytesIO(raw_bytes)) as image:
            if image.format not in ALLOWED_FORMATS:
                return error_response(
                    "UNSUPPORTED_FILE",
                    "Please upload a JPEG, PNG, or WebP image.",
                    415
                )

            image.verify()

        with Image.open(BytesIO(raw_bytes)) as image:
            image = image.convert("RGB")
            image = image.resize(
                IMAGE_SIZE,
                Image.Resampling.BILINEAR
            )

            image_array = np.asarray(
                image, dtype=np.float32
            )

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ):
        return error_response(
            "INVALID_IMAGE",
            "The uploaded file is not a valid image.",
            400
        )

    try:
        # The saved model includes MobileNetV2 preprocessing.
        # Do not apply preprocess_input a second time.
        batch = np.expand_dims(
            image_array, axis=0
        )

        cancer_score = float(
            model.predict(batch, verbose=0)[0][0]
        )

        if not np.isfinite(cancer_score):
            raise ValueError(
                "Model returned an invalid score."
            )

        cancer_score = float(
            np.clip(cancer_score, 0.0, 1.0)
        )

        if cancer_score >= 0.5:
            predicted_class = "Cancer"
            score = cancer_score
        else:
            predicted_class = "Non-Cancer"
            score = 1.0 - cancer_score

        # Generate Grad-CAM separately. If visualization fails,
        # still return the valid model prediction.
        grad_cam_url = None

        try:
            filename = generate_gradcam(
                image_array,
                cancer_score,
                predicted_class
            )

            grad_cam_url = url_for(
                "serve_gradcam",
                filename=filename,
                _external=True
            )

        except Exception:
            app.logger.exception(
                "Grad-CAM generation failed"
            )

        return jsonify({
            "prediction_id": f"pred_{uuid.uuid4().hex}",
            "predicted_class": predicted_class,
            "score": round(score, 4),
            "model_version": "v1",
            "grad_cam_url": grad_cam_url
        }), 200

    except Exception:
        app.logger.exception(
            "CNN inference failed"
        )

        return error_response(
            "PREDICTION_FAILED",
            "Unable to process this image.",
            500
        )


# -------------------------------
# Start the development server
# -------------------------------
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
