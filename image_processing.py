from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from config import IMAGE_SIZE, QUALITY_THRESHOLDS, QualityThresholds


@dataclass
class QualityMetrics:
    width: int
    height: int
    aspect_ratio: float
    brightness: float
    dark_clip_percent: float
    bright_clip_percent: float
    contrast: float
    blur_score: float
    noise_score: float


@dataclass
class ProcessingResult:
    original_rgb: np.ndarray
    processed_rgb: np.ndarray
    original_metrics: QualityMetrics
    processed_metrics: QualityMetrics
    detected_problems: list[str]
    applied_methods: list[str]
    final_status: str
    processing_time: float

    def report(self) -> dict:
        row = {f"before_{key}": value for key, value in asdict(self.original_metrics).items()}
        row.update({f"after_{key}": value for key, value in asdict(self.processed_metrics).items()})
        row.update(
            {
                "detected_problems": ", ".join(self.detected_problems) or "none",
                "applied_methods": ", ".join(self.applied_methods) or "none",
                "final_status": self.final_status,
                "processing_time": round(self.processing_time, 4),
            }
        )
        return row


def read_image(image_source) -> np.ndarray:
    """Read a path, uploaded bytes, or a NumPy image and return RGB uint8."""
    if isinstance(image_source, np.ndarray):
        image = image_source.copy()
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return np.clip(image, 0, 255).astype(np.uint8)

    if isinstance(image_source, (str, Path)):
        data = np.fromfile(str(image_source), dtype=np.uint8)
    elif isinstance(image_source, bytes):
        data = np.frombuffer(image_source, dtype=np.uint8)
    elif hasattr(image_source, "getvalue"):
        data = np.frombuffer(image_source.getvalue(), dtype=np.uint8)
    else:
        raise TypeError("Unsupported image source")

    decoded = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("The image is unreadable or corrupted")
    if decoded.ndim == 2:
        return cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)
    if decoded.shape[2] == 4:
        return cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def calculate_quality(image_rgb: np.ndarray) -> QualityMetrics:
    height, width = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    dark_clip_percent = float(np.mean(gray <= 5) * 100)
    bright_clip_percent = float(np.mean(gray >= 250) * 100)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    noise_score = float(np.std(gray.astype(np.float32) - smooth.astype(np.float32)))
    return QualityMetrics(
        width=width,
        height=height,
        aspect_ratio=round(width / max(height, 1), 4),
        brightness=round(brightness, 4),
        dark_clip_percent=round(dark_clip_percent, 4),
        bright_clip_percent=round(bright_clip_percent, 4),
        contrast=round(contrast, 4),
        blur_score=round(blur_score, 4),
        noise_score=round(noise_score, 4),
    )


def detect_problems(
    metrics: QualityMetrics,
    thresholds: QualityThresholds = QUALITY_THRESHOLDS,
) -> list[str]:
    problems = []
    if metrics.width < thresholds.reject_width or metrics.height < thresholds.reject_height:
        problems.append("very_low_resolution")
    elif metrics.width < thresholds.min_width or metrics.height < thresholds.min_height:
        problems.append("low_resolution")
    if metrics.brightness < thresholds.dark_brightness:
        problems.append("dark")
    elif metrics.brightness > thresholds.bright_brightness:
        problems.append("overexposed")
    if metrics.dark_clip_percent > thresholds.dark_clip_percent:
        problems.append("dark_clipping")
    if metrics.bright_clip_percent > thresholds.bright_clip_percent:
        problems.append("bright_clipping")
    if metrics.contrast < thresholds.low_contrast:
        problems.append("low_contrast")
    if metrics.blur_score < thresholds.severe_blur:
        problems.append("severe_blur")
    elif metrics.blur_score < thresholds.mild_blur:
        problems.append("mild_blur")
    if metrics.noise_score > thresholds.noise_score:
        problems.append("noise")
    return problems


def gamma_correction(image_rgb: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(float(gamma), 0.05)
    table = np.array([((value / 255.0) ** gamma) * 255 for value in range(256)]).astype("uint8")
    return cv2.LUT(image_rgb, table)


def improve_local_contrast(image_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((enhanced, channel_a, channel_b)), cv2.COLOR_LAB2RGB)


def reduce_noise(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(image_rgb, d=5, sigmaColor=35, sigmaSpace=35)


def unsharp_mask(image_rgb: np.ndarray, amount: float = 0.6) -> np.ndarray:
    blurred = cv2.GaussianBlur(image_rgb, (0, 0), 1.0)
    sharpened = cv2.addWeighted(image_rgb, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def resize_with_padding(
    image_rgb: np.ndarray,
    target_size: tuple[int, int] = IMAGE_SIZE,
    padding_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    target_height, target_width = target_size
    height, width = image_rgb.shape[:2]
    scale = min(target_width / width, target_height / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=interpolation)

    canvas = np.full((target_height, target_width, 3), padding_color, dtype=np.uint8)
    left = (target_width - new_width) // 2
    top = (target_height - new_height) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas


def smart_process_image(
    image_source,
    thresholds: QualityThresholds = QUALITY_THRESHOLDS,
    target_size: tuple[int, int] = IMAGE_SIZE,
) -> ProcessingResult:
    started = perf_counter()
    original = read_image(image_source)
    original_metrics = calculate_quality(original)
    problems = detect_problems(original_metrics, thresholds)
    processed = original.copy()
    methods = []

    if "very_low_resolution" not in problems:
        if "dark" in problems and "dark_clipping" not in problems:
            processed = gamma_correction(processed, 0.65)
            methods.append("gamma_brightening")
        elif "overexposed" in problems and "bright_clipping" not in problems:
            processed = gamma_correction(processed, 1.35)
            methods.append("gamma_darkening")

        if "low_contrast" in problems:
            processed = improve_local_contrast(processed)
            methods.append("clahe")
        if "noise" in problems:
            processed = reduce_noise(processed)
            methods.append("bilateral_filter")
        if "mild_blur" in problems:
            processed = unsharp_mask(processed)
            methods.append("unsharp_mask")

    if processed.shape[:2] != target_size:
        processed = resize_with_padding(processed, target_size)
        methods.append("aspect_ratio_resize_and_padding")

    processed_metrics = calculate_quality(processed)
    severe = {"very_low_resolution", "severe_blur", "dark_clipping", "bright_clipping"}
    if "very_low_resolution" in problems:
        status = "Rejected"
    elif severe.intersection(problems):
        status = "Review"
    elif methods:
        status = "Processed"
    else:
        status = "Accepted"

    return ProcessingResult(
        original_rgb=original,
        processed_rgb=processed,
        original_metrics=original_metrics,
        processed_metrics=processed_metrics,
        detected_problems=problems,
        applied_methods=methods,
        final_status=status,
        processing_time=perf_counter() - started,
    )


def save_rgb_image(path: str | Path, image_rgb: np.ndarray, quality: int = 95) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    extension = path.suffix.lower() or ".jpg"
    if extension in {".jpg", ".jpeg"}:
        success, encoded = cv2.imencode(extension, bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        success, encoded = cv2.imencode(extension, bgr)
    if not success:
        raise ValueError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def encode_jpeg(image_rgb: np.ndarray, quality: int = 95) -> bytes:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("Could not encode JPEG")
    return encoded.tobytes()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(image_source) -> str:
    import imagehash

    image_rgb = read_image(image_source)
    return str(imagehash.phash(Image.fromarray(image_rgb)))


def hamming_distance(hash_a: str, hash_b: str) -> int:
    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()


def quality_histogram(image_rgb: np.ndarray):
    import matplotlib.pyplot as plt

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    figure, axis = plt.subplots(figsize=(7, 3))
    axis.hist(gray.ravel(), bins=256, range=(0, 256), color="#087f5b")
    axis.set_title("Grayscale intensity histogram")
    axis.set_xlabel("Pixel intensity")
    axis.set_ylabel("Pixel count")
    axis.grid(alpha=0.15)
    figure.tight_layout()
    return figure
