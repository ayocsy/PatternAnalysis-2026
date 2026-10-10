"""Train and evaluate the ADNI CNN baseline or ConvNeXt classifier.

Evaluation is reported at three levels:

* slice: one prediction for each 2D JPEG;
* scan: mean AD probability across the 20 slices from one MRI scan;
* patient: mean AD probability across every included slice/scan for one patient.

Best checkpoints are selected using validation patient-level macro-F1 because
patients, rather than correlated slices, are the meaningful evaluation unit.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import Tensor, nn
from torch.utils.data import DataLoader

from dataset import ADNISliceDataset, SplitName, create_datasets
from modules import BaselineCNN, convnext_tiny, count_trainable_parameters


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


ModelName = Literal["baseline", "convnext"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an ADNI baseline CNN or ConvNeXt-Tiny model"
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip test evaluation during model development.",
    )
    parser.add_argument("--model", choices=("baseline", "convnext"), required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--profile-batches", type=int, default=20)
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
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def validate_args(arguments: argparse.Namespace) -> None:
    if arguments.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if arguments.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if arguments.workers < 0:
        raise ValueError("--workers cannot be negative")
    if arguments.patience <= 0:
        raise ValueError("--patience must be positive")
    if arguments.log_interval <= 0:
        raise ValueError("--log-interval must be positive")
    if arguments.profile_batches <= 0:
        raise ValueError("--profile-batches must be positive")

    for name in ("max_train_batches", "max_eval_batches"):
        value = getattr(arguments, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def set_reproducibility(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for repeatable experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker a deterministic but distinct seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_loaders(
    datasets: dict[SplitName, ADNISliceDataset],
    batch_size: int,
    workers: int,
    seed: int,
    pin_memory: bool,
) -> dict[SplitName, DataLoader]:
    """Construct deterministic loaders that share the same patient splits."""

    generator = torch.Generator()
    generator.manual_seed(seed)

    common_options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": workers > 0,
        "worker_init_fn": seed_worker,
    }

    return {
        "train": DataLoader(
            datasets["train"],
            shuffle=True,
            generator=generator,
            **common_options,
        ),
        "validation": DataLoader(
            datasets["validation"],
            shuffle=False,
            **common_options,
        ),
        "test": DataLoader(
            datasets["test"],
            shuffle=False,
            **common_options,
        ),
    }


def create_model(model_name: ModelName) -> nn.Module:
    if model_name == "baseline":
        return BaselineCNN(input_channels=1, num_classes=2)
    if model_name == "convnext":
        return convnext_tiny(input_channels=1, num_classes=2)
    raise ValueError(f"Unknown model: {model_name}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    log_interval: int,
    max_batches: int | None,
) -> dict[str, float]:
    """Run one optimization epoch and return slice-level training statistics."""

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

        if (batch_index + 1) % log_interval == 0:
            print(
                f"    batch {batch_index + 1:4d}/{len(loader):4d}  "
                f"loss={total_loss / total_samples:.4f}  "
                f"accuracy={total_correct / total_samples:.4f}",
                flush=True,
            )

    if total_samples == 0:
        raise RuntimeError("The training loader produced no samples")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "samples": float(total_samples),
    }


def aggregate_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
    group_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Average AD probabilities for each scan or patient."""

    grouped_probabilities: dict[str, list[float]] = defaultdict(list)
    grouped_labels: dict[str, int] = {}

    for label, probability, group_id in zip(labels, probabilities, group_ids):
        grouped_probabilities[group_id].append(float(probability))

        previous_label = grouped_labels.setdefault(group_id, int(label))
        if previous_label != int(label):
            raise RuntimeError(f"Group {group_id} contains inconsistent labels")

    ordered_group_ids = sorted(grouped_probabilities)
    aggregated_labels = np.asarray(
        [grouped_labels[group_id] for group_id in ordered_group_ids],
        dtype=np.int64,
    )
    aggregated_probabilities = np.asarray(
        [
            np.mean(grouped_probabilities[group_id])
            for group_id in ordered_group_ids
        ],
        dtype=np.float64,
    )

    return aggregated_labels, aggregated_probabilities


def compute_binary_metrics(
    labels: np.ndarray,
    ad_probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int | None]:
    """Compute per-class and aggregate binary classification metrics."""

    predictions = (ad_probabilities >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=(0, 1),
        zero_division=0,
    )

    auroc: float | None
    if len(np.unique(labels)) == 2:
        auroc = float(roc_auc_score(labels, ad_probabilities))
    else:
        auroc = None

    return {
        "examples": int(labels.shape[0]),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_nc": float(precision[0]),
        "recall_nc": float(recall[0]),
        "f1_nc": float(f1[0]),
        "precision_ad": float(precision[1]),
        "recall_ad": float(recall[1]),
        "f1_ad": float(f1[1]),
        "macro_f1": float((f1[0] + f1[1]) / 2.0),
        "auroc": auroc,
        "threshold": threshold,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    max_batches: int | None,
) -> dict[str, Any]:
    """Evaluate slice predictions and aggregate them by scan and patient."""

    model.eval()
    total_loss = 0.0
    total_samples = 0
    label_batches: list[Tensor] = []
    probability_batches: list[Tensor] = []
    scan_ids: list[str] = []
    patient_ids: list[str] = []

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        ad_probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        label_batches.append(labels.cpu())
        probability_batches.append(ad_probabilities.cpu())
        scan_ids.extend(batch["scan_id"])
        patient_ids.extend(batch["patient_id"])

    if total_samples == 0:
        raise RuntimeError("The evaluation loader produced no samples")

    labels = torch.cat(label_batches).numpy()
    probabilities = torch.cat(probability_batches).numpy()

    scan_labels, scan_probabilities = aggregate_probabilities(
        labels,
        probabilities,
        scan_ids,
    )
    patient_labels, patient_probabilities = aggregate_probabilities(
        labels,
        probabilities,
        patient_ids,
    )

    return {
        "loss": total_loss / total_samples,
        "slice": compute_binary_metrics(labels, probabilities),
        "scan": compute_binary_metrics(scan_labels, scan_probabilities),
        "patient": compute_binary_metrics(patient_labels, patient_probabilities),
    }


def format_metric(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def print_evaluation(name: str, results: dict[str, Any]) -> None:
    print(f"{name} loss: {results['loss']:.4f}")
    for level in ("slice", "scan", "patient"):
        metrics = results[level]
        print(
            f"  {level:7} "
            f"n={metrics['examples']:5d}  "
            f"accuracy={format_metric(metrics['accuracy'])}  "
            f"macro-F1={format_metric(metrics['macro_f1'])}  "
            f"AUROC={format_metric(metrics['auroc'])}"
        )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    model_name: ModelName,
    validation_results: dict[str, Any],
    arguments: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "validation_results": validation_results,
            "arguments": vars(arguments),
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Compatibility with PyTorch releases that predate weights_only.
        return torch.load(path, map_location=device)


def write_json(path: Path, value: Any) -> None:
    """Write JSON after converting non-finite floating-point values to null."""

    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): clean(nested) for key, nested in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(nested) for nested in item]
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, (np.floating, float)):
            number = float(item)
            return number if math.isfinite(number) else None
        if isinstance(item, np.integer):
            return int(item)
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(clean(value), file, indent=2, allow_nan=False)


def plot_history(history: list[dict[str, Any]], output_path: Path) -> None:
    epochs = [entry["epoch"] for entry in history]
    training_loss = [entry["train"]["loss"] for entry in history]
    validation_loss = [entry["validation"]["loss"] for entry in history]
    patient_macro_f1 = [
        entry["validation"]["patient"]["macro_f1"] for entry in history
    ]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, training_loss, label="Train")
    axes[0].plot(epochs, validation_loss, label="Validation")
    axes[0].set_title("Cross-entropy loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, patient_macro_f1, color="tab:green")
    axes[1].set_title("Validation patient macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


@torch.no_grad()
def profile_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    measured_batches: int,
) -> dict[str, float | int | None]:
    """Measure model-only inference latency and peak CUDA memory."""

    model.eval()
    warmup_batches = min(3, measured_batches)
    elapsed_seconds = 0.0
    measured_samples = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for batch_index, batch in enumerate(loader):
        if batch_index >= warmup_batches + measured_batches:
            break

        images = batch["image"].to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start_time = time.perf_counter()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            model(images)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        if batch_index >= warmup_batches:
            elapsed_seconds += time.perf_counter() - start_time
            measured_samples += images.shape[0]

    if measured_samples == 0:
        raise RuntimeError("Not enough test batches for inference profiling")

    peak_memory_mb: float | None = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    return {
        "measured_samples": measured_samples,
        "inference_latency_ms_per_slice": (
            elapsed_seconds * 1000.0 / measured_samples
        ),
        "throughput_slices_per_second": measured_samples / elapsed_seconds,
        "peak_inference_memory_mb": peak_memory_mb,
    }


def main() -> None:
    arguments = parse_args()
    validate_args(arguments)
    set_reproducibility(arguments.seed)

    model_name: ModelName = arguments.model
    device = select_device(arguments.device)
    use_amp = device.type == "cuda" and not arguments.no_amp

    if arguments.learning_rate is None:
        arguments.learning_rate = 1e-3 if model_name == "baseline" else 3e-4
    if arguments.weight_decay is None:
        arguments.weight_decay = 1e-4 if model_name == "baseline" else 5e-2
    if arguments.output_dir is None:
        arguments.output_dir = Path("outputs") / model_name

    output_dir: Path = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Experiment configuration")
    print("-" * 68)
    print("Model:", model_name)
    print("Device:", device)
    print("Mixed precision:", use_amp)
    print("Output directory:", output_dir)
    print("Seed:", arguments.seed)

    datasets = create_datasets(
        data_root=arguments.data_root,
        metadata_path=arguments.metadata_path,
        seed=arguments.seed,
        image_size=arguments.image_size,
        output_channels=1,
    )
    loaders = create_loaders(
        datasets=datasets,
        batch_size=arguments.batch_size,
        workers=arguments.workers,
        seed=arguments.seed,
        pin_memory=device.type == "cuda",
    )

    model = create_model(model_name).to(device)
    parameter_count = count_trainable_parameters(model)
    print(f"Trainable parameters: {parameter_count:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=arguments.epochs,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_path = output_dir / "best_model.pth"
    last_path = output_dir / "last_model.pth"
    history_path = output_dir / "history.json"
    curves_path = output_dir / "training_curves.png"
    history: list[dict[str, Any]] = []
    best_patient_macro_f1 = -math.inf
    epochs_without_improvement = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    training_start = time.perf_counter()

    for epoch in range(1, arguments.epochs + 1):
        print(f"\nEpoch {epoch:03d}/{arguments.epochs:03d}", flush=True)

        train_results = train_one_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            log_interval=arguments.log_interval,
            max_batches=arguments.max_train_batches,
        )
        validation_results = evaluate(
            model=model,
            loader=loaders["validation"],
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            max_batches=arguments.max_eval_batches,
        )
        scheduler.step()

        print(
            f"Train loss: {train_results['loss']:.4f}  "
            f"slice accuracy: {train_results['accuracy']:.4f}"
        )
        print_evaluation("Validation", validation_results)

        epoch_record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_results,
            "validation": validation_results,
        }
        history.append(epoch_record)
        write_json(history_path, history)
        plot_history(history, curves_path)

        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch,
            model_name,
            validation_results,
            arguments,
        )

        patient_macro_f1 = validation_results["patient"]["macro_f1"]
        if patient_macro_f1 > best_patient_macro_f1:
            best_patient_macro_f1 = patient_macro_f1
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                model_name,
                validation_results,
                arguments,
            )
            print("Saved new best patient-level checkpoint.")
        else:
            epochs_without_improvement += 1
            print(
                "No patient macro-F1 improvement for "
                f"{epochs_without_improvement} epoch(s)."
            )

        if epochs_without_improvement >= arguments.patience:
            print("Early stopping triggered.")
            break

    training_seconds = time.perf_counter() - training_start
    peak_training_memory_mb: float | None = None
    if device.type == "cuda":
        peak_training_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    checkpoint = load_checkpoint(best_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"\nLoaded best checkpoint from epoch {checkpoint['epoch']}.")
    if arguments.skip_test:
        print("Test evaluation skipped; profiling validation data.")
    else:
        test_results = evaluate(
            model=model,
            loader=loaders["test"],
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            max_batches=arguments.max_eval_batches,
        )
        print_evaluation("Test", test_results)
        write_json(output_dir / "test_metrics.json", test_results)

    inference_profile = profile_inference(
        model=model,
        loader=loaders["validation"],
        device=device,
        use_amp=use_amp,
        measured_batches=arguments.profile_batches,
    )
    resource_profile = {
        "model": model_name,
        "device": str(device),
        "trainable_parameters": parameter_count,
        "training_seconds": training_seconds,
        "peak_training_memory_mb": peak_training_memory_mb,
        **inference_profile,
    }

    write_json(output_dir / "resource_profile.json", resource_profile)
    write_json(output_dir / "configuration.json", vars(arguments))

    print("\nResource profile")
    print("-" * 68)
    print(f"Training time: {training_seconds:.2f} seconds")
    print(
        "Peak training memory: "
        f"{format_metric(peak_training_memory_mb)} MB"
    )
    print(
        "Inference latency: "
        f"{inference_profile['inference_latency_ms_per_slice']:.4f} ms/slice"
    )
    print(
        "Inference throughput: "
        f"{inference_profile['throughput_slices_per_second']:.2f} slices/s"
    )
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
