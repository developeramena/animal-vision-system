from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

CLASS_NAMES = [
    "butterfly",
    "cat",
    "chicken",
    "cow",
    "dog",
    "elephant",
    "horse",
    "sheep",
    "spider",
    "squirrel",
]

# Animals-10 uses Italian folder names. The second dataset uses English names.
CLASS_ALIASES = {
    "butterfly": {"butterfly", "farfalla", "butterflies"},
    "cat": {"cat", "gatto", "cats"},
    "chicken": {"chicken", "gallina", "chickens", "hen"},
    "cow": {"cow", "mucca", "cows", "cattle"},
    "dog": {"dog", "cane", "dogs"},
    "elephant": {"elephant", "elefante", "elephants"},
    "horse": {"horse", "cavallo", "horses"},
    "sheep": {"sheep", "pecora"},
    "spider": {"spider", "ragno", "spiders"},
    "squirrel": {"squirrel", "scoiattolo", "squirrels"},
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = (224, 224)
RANDOM_SEED = 123


@dataclass(frozen=True)
class QualityThresholds:
    min_width: int = 128
    min_height: int = 128
    reject_width: int = 64
    reject_height: int = 64
    dark_brightness: float = 55.0
    bright_brightness: float = 205.0
    low_contrast: float = 30.0
    mild_blur: float = 80.0
    severe_blur: float = 25.0
    dark_clip_percent: float = 15.0
    bright_clip_percent: float = 15.0
    noise_score: float = 12.0


@dataclass(frozen=True)
class TrainingSettings:
    image_height: int = IMAGE_SIZE[0]
    image_width: int = IMAGE_SIZE[1]
    channels: int = 3
    batch_size: int = 32
    feature_epochs: int = 12
    fine_tune_epochs: int = 20
    learning_rate: float = 0.001
    fine_tune_learning_rate: float = 0.00001
    dropout: float = 0.4
    dense_units: int = 256
    fine_tune_layers: int = 30


QUALITY_THRESHOLDS = QualityThresholds()
TRAINING_SETTINGS = TrainingSettings()

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.keras"
DEFAULT_CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "class_names.json"
DEFAULT_UNCERTAINTY_PATH = PROJECT_ROOT / "models" / "uncertainty_config.json"
