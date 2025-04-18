import os
import logging
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# ─── Configuration ─────────────────────────────────────────────────────────────
# Set paths to your dataset and output directories
DATA_DIR = "/kaggle/input/deepfake-and-real-images/Dataset"      # Should contain 'train/' and 'val/' subdirectories
OUTPUT_DIR = "/content/model"     # Directory to save models and logs

# Training parameters
IMG_SIZE = 299                     # Input image size for Xception
BATCH_SIZE = 32                    # Number of images per batch
EPOCHS = 50                        # Total number of training epochs
LEARNING_RATE = 1e-4               # Initial learning rate
MIXED_PRECISION_POLICY = "mixed_float16"  # Enables mixed precision for performance

# ─── Setup ─────────────────────────────────────────────────────────────────────
# Validate dataset directory
assert os.path.isdir(DATA_DIR), f"DATA_DIR not found: {DATA_DIR}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Enable mixed precision for performance optimization
mixed_precision.set_global_policy(MIXED_PRECISION_POLICY)

# ─── Model Definition ──────────────────────────────────────────────────────────
def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3)) -> Model:
    """
    Builds the deepfake detection model using Xception as the base.
    """
    base_model = Xception(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    base_model.trainable = False  # Freeze base model layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid", dtype="float32")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model

# ─── Data Generators ───────────────────────────────────────────────────────────
def create_data_generators():
    """
    Creates training and validation data generators with augmentation.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "Train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )
    val_generator = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "Validation"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )
    return train_generator, val_generator

# ─── Training ──────────────────────────────────────────────────────────────────
def train_and_fine_tune_model():
    """
    Compiles, trains, and fine-tunes the model, then saves the best version.
    """
    logging.info("Building the model...")
    model = build_model()

    logging.info("Preparing data generators...")
    train_gen, val_gen = create_data_generators()

    logging.info("Compiling the model...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    # Callbacks for initial training
    checkpoint_path = os.path.join(OUTPUT_DIR, "xception_model_v1.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

    logging.info("Starting initial training for %d epochs...", EPOCHS)
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Save the model after initial training
    saved_model_dir = os.path.join(OUTPUT_DIR, "saved_model")
    logging.info("Saving the model to %s", saved_model_dir)
    model.save(saved_model_dir)
    logging.info("Initial training complete.")

    # ─── Fine-Tuning ───────────────────────────────────────────────────────────
    logging.info("Unfreezing top layers of the base model for fine-tuning...")

    # Unfreeze the top N layers of the base model
    base_model = model.layers[1]  # Assuming base_model is the second layer
    for layer in base_model.layers[-20:]:  # Unfreeze last 20 layers
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    logging.info("Recompiling the model with a lower learning rate for fine-tuning...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    # Callbacks for fine-tuning
    fine_tune_checkpoint_path = os.path.join(OUTPUT_DIR, "xception_model_finetuned.h5")
    fine_tune_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=fine_tune_checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            verbose=1
        )
    ]

    logging.info("Starting fine-tuning...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Save the model after initial training
    saved_model_dir = os.path.join(OUTPUT_DIR, "saved_model")
    logging.info("Saving the model to %s", saved_model_dir)
    model.save(saved_model_dir)
    logging.info("Initial training complete.")

# ─── Main Execution ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_and_fine_tune_model()