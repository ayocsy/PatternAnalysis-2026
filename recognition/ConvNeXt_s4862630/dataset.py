"""Leakage-safe ADNI slice dataset for AD versus normal-control classification.

The course dataset arrives in ``train`` and ``test`` folders, but those folders
share patients. This module therefore discovers all images, recovers patient IDs
from the supplied metadata, and creates new patient-level splits.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import Callable, Literal, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


SplitName = Literal["train", "validation", "test"]

CLASS_TO_LABEL = {"NC": 0, "AD": 1}
METADATA_LABEL_TO_CLASS = {0: "NC", 2: "AD"}
PATIENT_PATTERN = re.compile(r"ADNI_(\d{3}_S_\d{4})")


@dataclass(frozen=True)
class SliceRecord:
    """Metadata needed to load and aggregate one 2D MRI slice."""

    path: Path
    label: int
    patient_id: str
    scan_id: str
    slice_index: int


def discover_slice_records(
    data_root: str | Path,
    metadata_path: str | Path,
) -> list[SliceRecord]:
    """Discover every JPEG and validate it against the supplied metadata.

    The original folder split is intentionally ignored. Only the ``AD`` and
    ``NC`` folder names are used as labels.
    """

    data_root = Path(data_root)
    metadata_path = Path(metadata_path)

    if not data_root.is_dir():
        raise FileNotFoundError(f"ADNI image directory not found: {data_root}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"ADNI metadata file not found: {metadata_path}")

    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    records: list[SliceRecord] = []
    scan_to_patient: dict[str, str] = {}
    patient_to_label: dict[str, int] = {}

    image_paths = sorted(data_root.glob("*/*/*.jpeg"))
    if not image_paths:
        raise RuntimeError(f"No JPEG images found below {data_root}")

    for image_path in image_paths:
        class_name = image_path.parent.name
        if class_name not in CLASS_TO_LABEL:
            raise ValueError(f"Unexpected class directory: {class_name}")

        try:
            scan_id, slice_text = image_path.stem.rsplit("_", 1)
            slice_index = int(slice_text)
        except ValueError as error:
            raise ValueError(f"Unexpected filename: {image_path.name}") from error

        if scan_id not in metadata:
            raise KeyError(f"Scan {scan_id} is missing from the metadata")

        entry = metadata[scan_id]
        metadata_class = METADATA_LABEL_TO_CLASS.get(entry.get("label"))
        if metadata_class != class_name:
            raise ValueError(
                f"Label mismatch for scan {scan_id}: folder={class_name}, "
                f"metadata={metadata_class}"
            )

        match = PATIENT_PATTERN.search(entry.get("raw", ""))
        if match is None:
            raise ValueError(f"Cannot extract patient ID for scan {scan_id}")

        patient_id = match.group(1)
        label = CLASS_TO_LABEL[class_name]

        previous_patient = scan_to_patient.setdefault(scan_id, patient_id)
        if previous_patient != patient_id:
            raise ValueError(f"Scan {scan_id} maps to multiple patients")

        previous_label = patient_to_label.setdefault(patient_id, label)
        if previous_label != label:
            raise ValueError(f"Patient {patient_id} has multiple class labels")

        records.append(
            SliceRecord(
                path=image_path,
                label=label,
                patient_id=patient_id,
                scan_id=scan_id,
                slice_index=slice_index,
            )
        )

    return records


def stratified_patient_split(
    records: Sequence[SliceRecord],
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 3710,
) -> dict[SplitName, list[SliceRecord]]:
    """Split records by patient while approximately preserving class ratios."""

    if not isclose(
        train_fraction + validation_fraction + test_fraction,
        1.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Split fractions must sum to 1.0")

    patient_to_label: dict[str, int] = {}
    patients_by_label: dict[int, list[str]] = defaultdict(list)

    for record in records:
        previous = patient_to_label.setdefault(record.patient_id, record.label)
        if previous != record.label:
            raise ValueError(f"Patient {record.patient_id} has multiple labels")

    for patient_id, label in patient_to_label.items():
        patients_by_label[label].append(patient_id)

    patient_split: dict[str, SplitName] = {}

    for label, patient_ids in sorted(patients_by_label.items()):
        patient_ids = sorted(patient_ids)
        random.Random(seed + label).shuffle(patient_ids)

        patient_count = len(patient_ids)
        train_count = round(patient_count * train_fraction)
        validation_count = round(patient_count * validation_fraction)
        test_count = patient_count - train_count - validation_count

        if min(train_count, validation_count, test_count) <= 0:
            raise ValueError(f"Class {label} has too few patients to split safely")

        train_end = train_count
        validation_end = train_end + validation_count

        for patient_id in patient_ids[:train_end]:
            patient_split[patient_id] = "train"
        for patient_id in patient_ids[train_end:validation_end]:
            patient_split[patient_id] = "validation"
        for patient_id in patient_ids[validation_end:]:
            patient_split[patient_id] = "test"

    split_records: dict[SplitName, list[SliceRecord]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for record in records:
        split_records[patient_split[record.patient_id]].append(record)

    validate_no_leakage(split_records)
    return split_records


def validate_no_leakage(
    split_records: dict[SplitName, Sequence[SliceRecord]],
) -> None:
    """Raise an error if any patient or scan appears in multiple splits."""

    patient_sets = {
        split: {record.patient_id for record in records}
        for split, records in split_records.items()
    }
    scan_sets = {
        split: {record.scan_id for record in records}
        for split, records in split_records.items()
    }

    split_names = list(split_records)
    for index, first in enumerate(split_names):
        for second in split_names[index + 1 :]:
            patient_overlap = patient_sets[first] & patient_sets[second]
            scan_overlap = scan_sets[first] & scan_sets[second]

            if patient_overlap:
                raise RuntimeError(
                    f"Patient leakage between {first} and {second}: "
                    f"{sorted(patient_overlap)[:5]}"
                )
            if scan_overlap:
                raise RuntimeError(
                    f"Scan leakage between {first} and {second}: "
                    f"{sorted(scan_overlap)[:5]}"
                )


def build_transform(
    training: bool,
    image_size: int = 224,
    output_channels: int = 1,
) -> transforms.Compose:
    """Create conservative transforms for structural brain MRI slices."""

    if output_channels not in (1, 3):
        raise ValueError("output_channels must be either 1 or 3")

    # The source images are 256x240 (width x height). Resize to 224x210 and
    # pad vertically to preserve anatomy without stretching it into a square.
    resized_height = round(image_size * 240 / 256)
    total_padding = image_size - resized_height
    top_padding = total_padding // 2
    bottom_padding = total_padding - top_padding

    operations: list[Callable] = [
        transforms.Grayscale(num_output_channels=output_channels),
        transforms.Resize((resized_height, image_size), antialias=True),
        transforms.Pad((0, top_padding, 0, bottom_padding), fill=0),
    ]

    if training:
        operations.append(
            transforms.RandomAffine(
                degrees=5,
                translate=(0.02, 0.02),
                scale=(0.98, 1.02),
            )
        )

    operations.append(transforms.ToTensor())

    if output_channels == 1:
        operations.append(transforms.Normalize(mean=(0.5,), std=(0.5,)))
    else:
        operations.append(
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            )
        )

    return transforms.Compose(operations)


class ADNISliceDataset(Dataset):
    """PyTorch dataset that returns a slice and its aggregation identifiers."""

    def __init__(
        self,
        records: Sequence[SliceRecord],
        transform: Callable | None = None,
    ) -> None:
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]

        with Image.open(record.path) as image:
            image = image.convert("L")
            if self.transform is not None:
                image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(record.label, dtype=torch.long),
            "patient_id": record.patient_id,
            "scan_id": record.scan_id,
            "slice_index": record.slice_index,
            "path": str(record.path),
        }


def create_datasets(
    data_root: str | Path = "/home/groups/comp3710/ADNI/AD_NC",
    metadata_path: str | Path = (
        "/home/groups/comp3710/ADNI/meta_data_with_label.json"
    ),
    seed: int = 3710,
    image_size: int = 224,
    output_channels: int = 1,
) -> dict[SplitName, ADNISliceDataset]:
    """Discover data, create leakage-safe splits, and construct datasets."""

    records = discover_slice_records(data_root, metadata_path)
    split_records = stratified_patient_split(records, seed=seed)

    return {
        "train": ADNISliceDataset(
            split_records["train"],
            build_transform(True, image_size, output_channels),
        ),
        "validation": ADNISliceDataset(
            split_records["validation"],
            build_transform(False, image_size, output_channels),
        ),
        "test": ADNISliceDataset(
            split_records["test"],
            build_transform(False, image_size, output_channels),
        ),
    }


def print_split_summary(datasets: dict[SplitName, ADNISliceDataset]) -> None:
    """Print slice, scan, patient, and patient-class counts for every split."""

    print("\nLeakage-safe ADNI split summary")
    print("-" * 68)

    for split_name, dataset in datasets.items():
        patient_labels: dict[str, int] = {}
        scans = set()

        for record in dataset.records:
            patient_labels[record.patient_id] = record.label
            scans.add(record.scan_id)

        class_counts = Counter(patient_labels.values())
        print(
            f"{split_name:10} "
            f"slices={len(dataset):5d}  "
            f"scans={len(scans):4d}  "
            f"patients={len(patient_labels):3d}  "
            f"NC={class_counts[0]:3d}  AD={class_counts[1]:3d}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the ADNI data pipeline")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/groups/comp3710/ADNI/AD_NC"),
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("/home/groups/comp3710/ADNI/meta_data_with_label.json"),
    )
    parser.add_argument("--seed", type=int, default=3710)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    adni_datasets = create_datasets(
        data_root=arguments.data_root,
        metadata_path=arguments.metadata_path,
        seed=arguments.seed,
    )
    print_split_summary(adni_datasets)

    sample = adni_datasets["train"][0]
    print("\nExample transformed sample")
    print("  Tensor shape:", tuple(sample["image"].shape))
    print("  Label:", sample["label"].item())
    print("  Patient:", sample["patient_id"])
    print("  Scan:", sample["scan_id"])
