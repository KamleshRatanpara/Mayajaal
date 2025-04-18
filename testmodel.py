import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import Xception, EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ─── Configuration ───────────────────────────────────────────────────────
DATA_DIR = "/dataset"  # Dataset root containing 'train' and 'val' folders
OUTPUT_DIR = "./models"
IMG_SIZE_XCEPTION = 299
IMG_SIZE_EFFICIENTNET = 300
BATCH_SIZE = 32
INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 10
MIXED_PRECISION_POLICY = "mixed_float16"

mixed_precision.set_global_policy(MIXED_PRECISION_POLICY)
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ─── Data Pipeline ────────────────────────────────────────────────
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

def get_generators(img_size):
    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "val"),
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )
    return train_gen, val_gen

# ─── Model Construction ────────────────────────────────────────────

def build_model(base_model_fn, input_shape, name, dropout_rate):
    base_model = base_model_fn(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(1, activation="sigmoid", dtype="float32")(x)

    model = Model(inputs=base_model.input, outputs=output, name=name)
    return model, base_model

# ─── Training and Fine-Tuning ────────────────────────────────────────

def train_and_finetune_model(model, base_model, train_gen, val_gen, model_name):
    checkpoint_path = os.path.join(OUTPUT_DIR, f"{model_name}.h5")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    callbacks = [
        ModelCheckpoint(filepath=checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
    ]

    logging.info(f"Initial training: {model_name}")
    model.fit(train_gen, validation_data=val_gen, epochs=INITIAL_EPOCHS, callbacks=callbacks)

    for layer in base_model.layers[-22:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    logging.info(f"Fine-tuning: {model_name}")
    model.fit(train_gen, validation_data=val_gen, epochs=FINE_TUNE_EPOCHS, callbacks=callbacks)
    model.save(os.path.join(OUTPUT_DIR, f"{model_name}_finetuned"))

# ─── Model Training ────────────────────────────────────────────
logging.info("Starting training for ensemble models")

x_train_gen, x_val_gen = get_generators(IMG_SIZE_XCEPTION)
e_train_gen, e_val_gen = get_generators(IMG_SIZE_EFFICIENTNET)

# Model 1: Xception, dropout 0.3
model1, base1 = build_model(Xception, (IMG_SIZE_XCEPTION, IMG_SIZE_XCEPTION, 3), "xception_A", dropout_rate=0.3)
train_and_finetune_model(model1, base1, x_train_gen, x_val_gen, "xception_A")

# Model 2: Xception, dropout 0.5
model2, base2 = build_model(Xception, (IMG_SIZE_XCEPTION, IMG_SIZE_XCEPTION, 3), "xception_B", dropout_rate=0.5)
train_and_finetune_model(model2, base2, x_train_gen, x_val_gen, "xception_B")

# Model 3: EfficientNetB3, dropout 0.3
model3, base3 = build_model(EfficientNetB3, (IMG_SIZE_EFFICIENTNET, IMG_SIZE_EFFICIENTNET, 3), "efficientnet_A", dropout_rate=0.3)
train_and_finetune_model(model3, base3, e_train_gen, e_val_gen, "efficientnet_A")

# Model 4: EfficientNetB3, dropout 0.5
model4, base4 = build_model(EfficientNetB3, (IMG_SIZE_EFFICIENTNET, IMG_SIZE_EFFICIENTNET, 3), "efficientnet_B", dropout_rate=0.5)
train_and_finetune_model(model4, base4, e_train_gen, e_val_gen, "efficientnet_B")

# ─── Ensemble Inference ────────────────────────────────────────────

def ensemble_predict(models, data_generator):
    predictions = [model.predict(data_generator, verbose=0) for model in models]
    avg_predictions = np.mean(predictions, axis=0)
    return (avg_predictions > 0.5).astype(int)

# Usage example (after training):
# loaded_models = [load_model(os.path.join(OUTPUT_DIR, f"{name}_finetuned")) for name in ["xception_A", "xception_B", "efficientnet_A", "efficientnet_B"]]
# final_predictions = ensemble_predict(loaded_models, val_generator)
# print(final_predictions)
