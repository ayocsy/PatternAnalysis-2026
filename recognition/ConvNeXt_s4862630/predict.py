"""Demonstrate saved ADNI models on selected held-out patients.

Loads checkpoint weights and validation-fitted calibration settings. No fitting,
training or threshold selection takes place here. Patient predictions average
all included slice probabilities, matching train.py and evaluate_models.py.

Example:
python predict.py --checkpoint outputs/convnext_seed3710/best_model.pth \
    --calibration outputs/evaluation_convnext_seed3710/calibration.json \
    --patient-id 168_S_6086 --output-dir outputs/demo_convnext
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from scipy.special import expit

from evaluate_models import (
    checkpoint_hash, group_predictions, predict, save_csv, save_json, split_manifest,
)


def summarize(rows, margins, settings):
    """Apply the saved thresholds at their matching aggregation levels."""
    raw = expit(margins)
    calibrated = expit(margins / settings["temperature"])
    results = {}
    for level in ("slice", "scan", "patient"):
        ids, labels, probabilities = group_predictions(rows, calibrated, level)
        _, _, raw_p = group_predictions(rows, raw, level)
        threshold = settings["thresholds"][level]
        predictions = (probabilities >= threshold).astype(int)
        records = []
        for identifier, label, probability, original_p, prediction in zip(
            ids, labels, probabilities, raw_p, predictions,
        ):
            confidence = float(probability if prediction else 1 - probability)
            record = {
                "id": identifier, "true_class": "AD" if label else "NC",
                "predicted_class": "AD" if prediction else "NC",
                "raw_ad_probability": float(original_p),
                "calibrated_ad_probability": float(probability),
                "decision_threshold": float(threshold),
                "assigned_class_probability": confidence,
                "correct": bool(prediction == label),
            }
            if level == "patient":
                cutoff = settings["patient_rejection"]["reject_threshold"]
                record["reject_threshold"] = float(cutoff)
                record["experimental_rule_action"] = (
                    "review" if confidence < cutoff else "accept_prediction"
                )
            records.append(record)
        results[level] = records
    details = [dict(row, logit_margin=float(margin),
                    raw_ad_probability=float(rp), calibrated_ad_probability=float(cp))
               for row, margin, rp, cp in zip(rows, margins, raw, calibrated)]
    return results, details


def plot_demo(path, rows, results, model_name):
    """Show three evenly spaced slices per patient; these are not heatmaps."""
    patients = results["patient"]
    fig, axes = plt.subplots(len(patients), 3, figsize=(11, 3.5 * len(patients)), squeeze=False)
    for index, patient in enumerate(patients):
        samples = sorted(
            [r for r in rows if r["patient_id"] == patient["id"]],
            key=lambda r: (r["scan_id"], r["slice_index"]),
        )
        selected = np.linspace(0, len(samples) - 1, 3).round().astype(int)
        for column, sample_index in enumerate(selected):
            sample = samples[sample_index]
            ax = axes[index, column]
            with Image.open(sample["path"]) as image:
                ax.imshow(np.asarray(image.convert("L")), cmap="gray")
            ax.set_title(f"Scan {sample['scan_id']}, slice {sample['slice_index']}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if column == 0:
                ax.set_ylabel(
                    f"{patient['id']}\nTrue {patient['true_class']}; predicted {patient['predicted_class']}\n"
                    f"Patient P(AD)={patient['calibrated_ad_probability']:.3f}\n"
                    f"Rule: {patient['experimental_rule_action']}", fontsize=9,
                )
    fig.suptitle(f"{model_name}: patient prediction demonstration", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patient-id", action="append", help="Repeat to demonstrate multiple patients")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--max-patients", type=int, default=3,
                        help="Number shown when no patient IDs are supplied")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--metadata-path", type=Path)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers < 0 or args.max_patients <= 0:
        parser.error("Invalid batch size, worker count or patient count")
    if (args.output_dir / "predictions.json").exists():
        parser.error("Demo output already exists; choose a new output directory")

    from dataset import ADNISliceDataset, create_datasets
    from train import create_model, load_checkpoint, select_device, set_reproducibility
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    saved = checkpoint["arguments"]
    settings = json.loads(args.calibration.read_text())
    set_reproducibility(int(saved["seed"]))
    datasets = create_datasets(
        data_root=args.data_root or saved["data_root"],
        metadata_path=args.metadata_path or saved["metadata_path"],
        seed=int(saved["seed"]), image_size=int(saved["image_size"]), output_channels=1,
    )
    amp = device.type == "cuda" and not saved.get("no_amp", False)
    identity = {
        "checkpoint_sha256": checkpoint_hash(args.checkpoint),
        "model": checkpoint["model_name"], "epoch": int(checkpoint["epoch"]),
        "seed": int(saved["seed"]), "image_size": int(saved["image_size"]),
        "splits": split_manifest(datasets), "mixed_precision": amp,
    }
    if identity != settings["identity"]:
        raise RuntimeError("Checkpoint, dataset split or precision does not match calibration")
    source = datasets[args.split]
    available = sorted({r.patient_id for r in source.records})
    selected = sorted(set(args.patient_id)) if args.patient_id else available[:args.max_patients]
    missing = set(selected) - set(available)
    if missing:
        parser.error(f"Patients not present in {args.split}: {sorted(missing)}")
    records = [r for r in source.records if r.patient_id in selected]
    subset = ADNISliceDataset(records, transform=source.transform)
    model = create_model(checkpoint["model_name"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model={identity['model']} epoch={identity['epoch']} patients={selected}", flush=True)
    rows, margins = predict(model, subset, args.batch_size, args.workers, device, amp)
    results, details = summarize(rows, margins, settings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "predictions.json", {
        "identity": identity, "split": args.split, "temperature": settings["temperature"],
        "patient_ids": selected, "predictions": results,
        "note": "Selected-case demonstration, not a new full-test benchmark. No parameters fitted.",
    })
    for level, values in results.items():
        save_csv(args.output_dir / f"{level}_predictions.csv", values)
    save_csv(args.output_dir / "slice_details.csv", details)
    plot_demo(args.output_dir / "prediction_examples.png", rows, results, identity["model"])
    for patient in results["patient"]:
        print(f"Patient {patient['id']} | true={patient['true_class']} "
              f"predicted={patient['predicted_class']} "
              f"P(AD)={patient['calibrated_ad_probability']:.4f} "
              f"action={patient['experimental_rule_action']}", flush=True)
    print(f"Saved demonstration to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
