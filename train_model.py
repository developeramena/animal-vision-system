from argparse import ArgumentParser
from pathlib import Path
import json

from config import TRAINING_SETTINGS
from model_utils import (
    build_efficientnet_model,
    calibrate_confidence_threshold,
    compile_feature_extraction,
    create_datasets,
    evaluate_model,
    plot_training_history,
    prepare_fine_tuning,
    training_callbacks,
)


def train_complete_model(split_root: Path, output_directory: Path):
    output_directory.mkdir(parents=True, exist_ok=True)
    feature_directory = output_directory / "feature_extraction"
    train_ds, val_ds, test_ds, class_names = create_datasets(split_root)

    model = build_efficientnet_model(len(class_names))
    compile_feature_extraction(model)
    model.summary()

    history_feature = model.fit(
        train_ds,
        epochs=TRAINING_SETTINGS.feature_epochs,
        validation_data=val_ds,
        callbacks=training_callbacks(feature_directory),
    )

    prepare_fine_tuning(model)
    history_fine = model.fit(
        train_ds,
        epochs=TRAINING_SETTINGS.feature_epochs + TRAINING_SETTINGS.fine_tune_epochs,
        initial_epoch=len(history_feature.epoch),
        validation_data=val_ds,
        callbacks=training_callbacks(output_directory),
    )

    model.save(output_directory / "best_model.keras")
    (output_directory / "class_names.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )
    (output_directory / "model_config.json").write_text(
        json.dumps(
            {
                "architecture": "EfficientNetB0",
                "input_size": [224, 224, 3],
                "class_count": len(class_names),
                "class_names": class_names,
                "preprocessing": "RGB uint8 in 0-255 range; EfficientNet internal rescaling",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_training_history(
        [history_feature, history_fine], output_directory / "training_history.png"
    )
    calibrate_confidence_threshold(
        model, val_ds, output_directory / "uncertainty_config.json"
    )
    metrics, report, figure = evaluate_model(
        model, test_ds, class_names, output_directory
    )
    print("Final test metrics:", metrics)
    return model, metrics


def main():
    parser = ArgumentParser(description="Train the Animal Vision EfficientNetB0 model")
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--output", default=Path("models"), type=Path)
    arguments = parser.parse_args()
    train_complete_model(arguments.split_root, arguments.output)


if __name__ == "__main__":
    main()
