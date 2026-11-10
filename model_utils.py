from pathlib import Path
import json

import cv2
import numpy as np

from config import IMAGE_SIZE, TRAINING_SETTINGS, TrainingSettings


def create_datasets(split_root: str | Path, settings: TrainingSettings = TRAINING_SETTINGS):
    import tensorflow as tf

    split_root = Path(split_root)
    common = {
        "image_size": (settings.image_height, settings.image_width),
        "color_mode": "rgb",
        "batch_size": settings.batch_size,
    }
    train_ds = tf.keras.utils.image_dataset_from_directory(
        split_root / "train", shuffle=True, seed=123, **common
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        split_root / "val", shuffle=False, **common
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        split_root / "test", shuffle=False, **common
    )
    class_names = train_ds.class_names
    if val_ds.class_names != class_names or test_ds.class_names != class_names:
        raise ValueError("Class order differs between train, validation, and test folders")
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names


def build_efficientnet_model(
    number_of_classes: int,
    settings: TrainingSettings = TRAINING_SETTINGS,
):
    import tensorflow as tf

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(settings.image_height, settings.image_width, settings.channels),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(
        shape=(settings.image_height, settings.image_width, settings.channels),
        name="input_image",
    )
    x = augmentation(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.BatchNormalization(name="batch_normalization")(x)
    x = tf.keras.layers.Dense(settings.dense_units, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(settings.dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        number_of_classes, activation="softmax", name="predictions"
    )(x)
    model = tf.keras.Model(inputs, outputs, name="animal_efficientnetb0")
    return model


def compile_feature_extraction(model, settings: TrainingSettings = TRAINING_SETTINGS):
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=settings.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def prepare_fine_tuning(
    model,
    settings: TrainingSettings = TRAINING_SETTINGS,
):
    import tensorflow as tf

    base_model = model.get_layer("efficientnetb0")
    base_model.trainable = True
    freeze_until = max(0, len(base_model.layers) - settings.fine_tune_layers)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    for layer in base_model.layers[freeze_until:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=settings.fine_tune_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def training_callbacks(output_directory: str | Path):
    import tensorflow as tf

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            output_directory / "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(output_directory / "training_history.csv"),
    ]


def plot_training_history(histories, output_path: str | Path):
    import matplotlib.pyplot as plt

    accuracy, validation_accuracy, loss, validation_loss = [], [], [], []
    for history in histories:
        accuracy.extend(history.history.get("accuracy", []))
        validation_accuracy.extend(history.history.get("val_accuracy", []))
        loss.extend(history.history.get("loss", []))
        validation_loss.extend(history.history.get("val_loss", []))
    epochs = range(1, len(accuracy) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, accuracy, marker="o", label="Training Accuracy")
    axes[0].plot(epochs, validation_accuracy, marker="o", label="Validation Accuracy")
    axes[0].set_title("Training and Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.2)
    axes[0].legend()
    axes[1].plot(epochs, loss, marker="o", label="Training Loss")
    axes[1].plot(epochs, validation_loss, marker="o", label="Validation Loss")
    axes[1].set_title("Training and Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    return figure


def evaluate_model(model, test_dataset, class_names, output_directory: str | Path):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    probabilities = model.predict(test_dataset, verbose=1)
    predicted_labels = np.argmax(probabilities, axis=1)
    true_labels = np.concatenate([labels.numpy() for _, labels in test_dataset])
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_directory / "classification_report.csv")
    matrix = confusion_matrix(true_labels, predicted_labels)
    figure, axis = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_title("EfficientNetB0 Confusion Matrix")
    figure.tight_layout()
    figure.savefig(output_directory / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    metrics = {"test_loss": float(test_loss), "test_accuracy": float(test_accuracy)}
    (output_directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics, report, figure


def calibrate_confidence_threshold(
    model,
    validation_dataset,
    output_path: str | Path,
    candidate_thresholds=None,
):
    """Select an uncertainty threshold using correct and incorrect validation predictions."""
    if candidate_thresholds is None:
        candidate_thresholds = np.arange(0.40, 0.91, 0.05)
    probabilities = model.predict(validation_dataset, verbose=0)
    predicted = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    true_labels = np.concatenate([labels.numpy() for _, labels in validation_dataset])
    correct = predicted == true_labels

    results = []
    for threshold in candidate_thresholds:
        accepted = confidence >= threshold
        correct_acceptance = float(np.mean(accepted[correct])) if np.any(correct) else 0.0
        error_rejection = float(np.mean(~accepted[~correct])) if np.any(~correct) else 1.0
        balanced_score = (correct_acceptance + error_rejection) / 2.0
        results.append(
            {
                "threshold": round(float(threshold), 4),
                "correct_acceptance": correct_acceptance,
                "error_rejection": error_rejection,
                "balanced_score": balanced_score,
            }
        )
    best = max(results, key=lambda row: row["balanced_score"])
    configuration = {
        "confidence_threshold": best["threshold"],
        "selection_method": "validation_correctness_balanced_score",
        "limitation": (
            "This threshold measures uncertainty for known classes. Add outside-class "
            "images before claiming unknown-animal detection."
        ),
        "candidates": results,
    }
    Path(output_path).write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    return configuration


def load_class_names(path: str | Path) -> list[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def predict_top_k(model, image_rgb: np.ndarray, class_names: list[str], top_k: int = 3):
    image = cv2.resize(image_rgb, IMAGE_SIZE[::-1], interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(image.astype(np.float32), axis=0)
    probabilities = model.predict(batch, verbose=0)[0]
    indices = np.argsort(probabilities)[::-1][:top_k]
    return [
        {"class_name": class_names[index], "probability": float(probabilities[index])}
        for index in indices
    ]


def make_gradcam_heatmap(model, image_rgb: np.ndarray, class_index: int | None = None):
    import tensorflow as tf

    base_model = model.get_layer("efficientnetb0")
    last_conv_layer = base_model.get_layer("top_conv")
    convolution_model = tf.keras.Model(
        base_model.inputs,
        [last_conv_layer.output, base_model.output],
    )
    image = cv2.resize(image_rgb, IMAGE_SIZE[::-1], interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(image.astype(np.float32), axis=0)
    augmented = model.get_layer("augmentation")(batch, training=False)

    with tf.GradientTape() as tape:
        convolution_output, base_output = convolution_model(augmented, training=False)
        x = model.get_layer("global_average_pooling")(base_output)
        x = model.get_layer("batch_normalization")(x, training=False)
        x = model.get_layer("dense_features")(x)
        x = model.get_layer("dropout")(x, training=False)
        predictions = model.get_layer("predictions")(x)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, convolution_output)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(convolution_output[0] * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, heatmap)
    return heatmap.numpy()


def overlay_gradcam(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4):
    resized = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 1.0 - alpha, colored, alpha, 0)
    return np.clip(overlay, 0, 255).astype(np.uint8)
