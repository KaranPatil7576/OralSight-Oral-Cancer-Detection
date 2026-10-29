
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "oral_cancer.keras"
TEST_DIR = BASE_DIR / "dataset" / "test"

CLASS_NAMES = ["Non-Cancer", "Cancer"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8

model = tf.keras.models.load_model(MODEL_PATH)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

y_true = []
y_scores = []

for images, labels in test_ds:
    scores = model.predict(images, verbose=0).reshape(-1)
    y_true.extend(labels.numpy().reshape(-1).astype(int))
    y_scores.extend(scores)

y_true = np.array(y_true)
y_scores = np.array(y_scores)
y_pred = (y_scores >= 0.5).astype(int)

print("\nClass mapping:", CLASS_NAMES)
print("Model output shape:", model.output_shape)
print("\nConfusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=CLASS_NAMES,
    zero_division=0,
))

print("\nIndividual predictions:")
for path, actual, score in zip(
    test_ds.file_paths, y_true, y_scores
):
    predicted = int(score >= 0.5)
    print(
        f"{Path(path).parent.name:12} | "
        f"predicted={CLASS_NAMES[predicted]:12} | "
        f"Cancer score={score:.4f} | "
        f"actual={CLASS_NAMES[actual]} | "
        f"{Path(path).name}"
    )
