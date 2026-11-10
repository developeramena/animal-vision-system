from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import json
import shutil

import pandas as pd

from config import CLASS_ALIASES, CLASS_NAMES, IMAGE_EXTENSIONS, RANDOM_SEED
from image_processing import (
    file_sha256,
    hamming_distance,
    perceptual_hash,
    save_rgb_image,
    smart_process_image,
)


@dataclass(frozen=True)
class DatasetSource:
    name: str
    root: Path


def normalize_label(folder_name: str) -> str | None:
    label = folder_name.strip().lower().replace("-", "_").replace(" ", "_")
    for class_name, aliases in CLASS_ALIASES.items():
        normalized_aliases = {alias.replace("-", "_").replace(" ", "_") for alias in aliases}
        if label in normalized_aliases:
            return class_name
    return None


def find_class_directories(root: str | Path) -> dict[str, list[Path]]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    found = defaultdict(list)
    for directory in [root, *root.rglob("*")]:
        if not directory.is_dir():
            continue
        class_name = normalize_label(directory.name)
        if class_name and any(
            child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
            for child in directory.iterdir()
        ):
            found[class_name].append(directory)
    return dict(found)


def list_source_images(source: DatasetSource) -> dict[str, list[Path]]:
    class_directories = find_class_directories(source.root)
    images = {class_name: [] for class_name in CLASS_NAMES}
    for class_name in CLASS_NAMES:
        for directory in class_directories.get(class_name, []):
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    images[class_name].append(path)
    return images


def interleave_sources(source_lists: list[list[tuple[str, Path]]]):
    queues = [deque(items) for items in source_lists if items]
    while queues:
        next_round = []
        for queue in queues:
            if queue:
                yield queue.popleft()
            if queue:
                next_round.append(queue)
        queues = next_round


class NearDuplicateIndex:
    """Find pHash candidates efficiently by indexing eight one-byte chunks."""

    def __init__(self, maximum_distance: int = 5):
        self.maximum_distance = maximum_distance
        self.hashes: list[tuple[str, str]] = []
        self.buckets: dict[tuple[int, int], set[int]] = defaultdict(set)

    @staticmethod
    def chunks(hash_text: str):
        value = int(hash_text, 16)
        for index in range(8):
            yield index, (value >> (index * 8)) & 0xFF

    def find(self, hash_text: str) -> tuple[str, int] | None:
        candidates = set()
        for key in self.chunks(hash_text):
            candidates.update(self.buckets.get(key, set()))
        for index in candidates:
            image_id, known_hash = self.hashes[index]
            distance = hamming_distance(hash_text, known_hash)
            if distance <= self.maximum_distance:
                return image_id, distance
        return None

    def add(self, image_id: str, hash_text: str) -> None:
        index = len(self.hashes)
        self.hashes.append((image_id, hash_text))
        for key in self.chunks(hash_text):
            self.buckets[key].add(index)


def build_clean_dataset(
    sources: list[DatasetSource],
    output_root: str | Path,
    target_per_class: int = 500,
    near_duplicate_distance: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge two sources, process images, remove duplicates, and write reports."""
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output already exists: {output_root}. Use a new dataset version folder."
        )
    cleaned_root = output_root / "cleaned"
    review_root = output_root / "review"
    quarantine_root = output_root / "quarantine"
    reports_root = output_root / "reports"
    for directory in (cleaned_root, review_root, quarantine_root, reports_root):
        directory.mkdir(parents=True, exist_ok=True)

    source_images = {source.name: list_source_images(source) for source in sources}
    manifest_rows = []
    quality_rows = []
    exact_hashes: dict[str, str] = {}
    near_index = NearDuplicateIndex(near_duplicate_distance)
    accepted_counts = {class_name: 0 for class_name in CLASS_NAMES}

    for class_name in CLASS_NAMES:
        per_source = []
        for source in sources:
            items = [(source.name, path) for path in source_images[source.name][class_name]]
            per_source.append(items)

        sequence = 0
        for source_name, original_path in interleave_sources(per_source):
            if accepted_counts[class_name] >= target_per_class:
                break
            sequence += 1
            image_id = f"{class_name}_{sequence:05d}"
            manifest_rows.append(
                {
                    "image_id": image_id,
                    "source_dataset": source_name,
                    "class_name": class_name,
                    "original_path": str(original_path),
                    "original_filename": original_path.name,
                }
            )

            base_row = {
                "image_id": image_id,
                "source_dataset": source_name,
                "class_name": class_name,
                "original_path": str(original_path),
            }
            try:
                exact_hash = file_sha256(original_path)
                if exact_hash in exact_hashes:
                    quality_rows.append(
                        {
                            **base_row,
                            "sha256": exact_hash,
                            "duplicate_group": exact_hashes[exact_hash],
                            "final_status": "Rejected Duplicate",
                            "notes": "Exact duplicate",
                        }
                    )
                    continue

                visual_hash = perceptual_hash(original_path)
                near_match = near_index.find(visual_hash)
                if near_match:
                    duplicate_id, distance = near_match
                    quality_rows.append(
                        {
                            **base_row,
                            "sha256": exact_hash,
                            "perceptual_hash": visual_hash,
                            "duplicate_group": duplicate_id,
                            "final_status": "Review Duplicate",
                            "notes": f"Near duplicate candidate; Hamming distance {distance}",
                        }
                    )
                    continue

                result = smart_process_image(original_path)
                report = result.report()
                output_path = ""
                if result.final_status in {"Accepted", "Processed"}:
                    accepted_counts[class_name] += 1
                    filename = f"{source_name}_{class_name}_{accepted_counts[class_name]:05d}.jpg"
                    destination = cleaned_root / class_name / filename
                    save_rgb_image(destination, result.processed_rgb)
                    output_path = str(destination)
                elif result.final_status == "Review":
                    filename = f"{source_name}_{class_name}_{sequence:05d}.jpg"
                    destination = review_root / class_name / filename
                    save_rgb_image(destination, result.processed_rgb)
                    output_path = str(destination)

                exact_hashes[exact_hash] = image_id
                near_index.add(image_id, visual_hash)
                quality_rows.append(
                    {
                        **base_row,
                        **report,
                        "sha256": exact_hash,
                        "perceptual_hash": visual_hash,
                        "duplicate_group": "",
                        "output_path": output_path,
                        "notes": "",
                    }
                )
            except Exception as error:
                quarantine_path = quarantine_root / class_name / original_path.name
                quarantine_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(original_path, quarantine_path)
                except OSError:
                    pass
                quality_rows.append(
                    {
                        **base_row,
                        "final_status": "Quarantined",
                        "output_path": str(quarantine_path),
                        "notes": str(error),
                    }
                )

    manifest = pd.DataFrame(manifest_rows)
    quality = pd.DataFrame(quality_rows)
    manifest.to_csv(reports_root / "source_manifest.csv", index=False)
    quality.to_csv(reports_root / "image_quality.csv", index=False)

    if quality.empty:
        summary = pd.DataFrame(columns=["class_name", "final_status", "count"])
    else:
        summary = (
            quality.groupby(["class_name", "final_status"], dropna=False)
            .size()
            .reset_index(name="count")
        )
    summary.to_csv(reports_root / "dataset_summary.csv", index=False)

    configuration = {
        "classes": CLASS_NAMES,
        "target_per_class": target_per_class,
        "near_duplicate_distance": near_duplicate_distance,
        "sources": [{"name": source.name, "root": str(source.root)} for source in sources],
    }
    (reports_root / "dataset_config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    return quality, summary


def split_cleaned_dataset(
    cleaned_root: str | Path,
    split_root: str | Path,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split

    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("Train, validation, and test ratios must total 1.0")
    cleaned_root = Path(cleaned_root)
    split_root = Path(split_root)
    if split_root.exists() and any(split_root.iterdir()):
        raise FileExistsError(f"Split output already exists: {split_root}")

    rows = []
    for class_name in CLASS_NAMES:
        class_directory = cleaned_root / class_name
        images = sorted(
            path for path in class_directory.glob("*")
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(images) < 10:
            raise ValueError(f"Class {class_name} needs at least 10 cleaned images")

        train_files, temporary_files = train_test_split(
            images,
            test_size=1.0 - train_ratio,
            random_state=random_seed,
            shuffle=True,
        )
        test_share = test_ratio / (validation_ratio + test_ratio)
        validation_files, test_files = train_test_split(
            temporary_files,
            test_size=test_share,
            random_state=random_seed,
            shuffle=True,
        )

        for split_name, files in (
            ("train", train_files),
            ("val", validation_files),
            ("test", test_files),
        ):
            destination_directory = split_root / split_name / class_name
            destination_directory.mkdir(parents=True, exist_ok=True)
            for source_path in files:
                destination = destination_directory / source_path.name
                shutil.copy2(source_path, destination)
                rows.append(
                    {
                        "class_name": class_name,
                        "split": split_name,
                        "source_path": str(source_path),
                        "split_path": str(destination),
                    }
                )

    split_manifest = pd.DataFrame(rows)
    split_manifest.to_csv(split_root / "split_manifest.csv", index=False)
    return split_manifest
