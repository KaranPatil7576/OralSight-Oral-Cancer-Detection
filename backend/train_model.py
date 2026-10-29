
from pathlib import Path
import json
import tensorflow as tf

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 15
SEED = 42

# Output index 0 = Non-Cancer
# Output index 1 = Cancer
CLASS_NAMES = ["Non-Cancer", "Cancer"]

# Check required directories
for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Missing folder: {split_dir}"
        )

    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class folder: {class_dir}"
            )

        image_count = sum(
            1 for p in class_dir.iterdir()
            if p.is_file()
        )

        print(f"{split_dir.name}/{class_name}: "
              f"{image_count} files")

        if image_count == 0:
            raise ValueError(
                f"No images found in {class_dir}"
            )

# Load datasets
def load_dataset(directory, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=shuffle,
        seed=SEED,
    )

train_ds = load_dataset(TRAIN_DIR, True)
val_ds = load_dataset(VAL_DIR, False)
test_ds = load_dataset(TEST_DIR, False)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# Data augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
])

# Pretrained CNN
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = augmentation(inputs)

# MobileNetV2 preprocessing: [0, 255] -> [-1, 1]
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    1, activation="sigmoid"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

model.summary()

model_path = MODEL_DIR / "oral_cancer.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(model_path),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
    ),
]

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# Evaluate the saved best model on held-out test data
best_model = tf.keras.models.load_model(model_path)

print("\nTest set evaluation:")
best_model.evaluate(test_ds, return_dict=True)

# Save class mapping and model information
metadata = {
    "class_names": CLASS_NAMES,
    "image_size": list(IMAGE_SIZE),
    "architecture": "MobileNetV2",
    "model_version": "v1",
}

with open(MODEL_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nModel saved at: {model_path}")
print("Training completed.")
    