"""Evaluate saved ADNI checkpoints without retraining.

calibrate: read validation images only; fit one temperature using patient NLL,
choose thresholds using validation macro-F1, and freeze settings in JSON.
test: verify checkpoint/split identity and apply frozen settings to test images.

Patient aggregation matches train.py: average probabilities over all slices.
Calibration and threshold selection reuse the checkpoint-selection validation
set, so validation scores are development results, not unbiased estimates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, roc_curve,
)

LEVELS = ("slice", "scan", "patient")
METHODS = ("raw_0.5", "temperature_0.5", "temperature_tuned")


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def checkpoint_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_manifest(datasets):
    # Exclude absolute paths so a relocated, otherwise identical dataset works.
    manifest = {}
    for name, dataset in datasets.items():
        records = sorted(
            (r.patient_id, r.scan_id, r.slice_index, r.label)
            for r in dataset.records
        )
        manifest[name] = {
            "slices": len(records),
            "patients": len({r[0] for r in records}),
            "sha256": hashlib.sha256(json.dumps(records).encode()).hexdigest(),
        }
    return manifest


def group_predictions(rows, probabilities, level):
    labels = np.array([r["label"] for r in rows], dtype=np.int64)
    if len(rows) != len(probabilities) or not len(rows):
        raise ValueError("Predictions must match a nonempty set of rows")
    if level == "slice":
        ids = [f'{r["scan_id"]}_{r["slice_index"]}' for r in rows]
        return ids, labels, np.asarray(probabilities, dtype=np.float64)
    key = "scan_id" if level == "scan" else "patient_id"
    groups = {}
    for row, probability in zip(rows, probabilities):
        group = groups.setdefault(row[key], {"label": row["label"], "values": []})
        if group["label"] != row["label"]:
            raise ValueError("Inconsistent labels in " + row[key])
        group["values"].append(float(probability))
    ids = sorted(groups)
    return (
        ids,
        np.array([groups[i]["label"] for i in ids], dtype=np.int64),
        np.array([np.mean(groups[i]["values"]) for i in ids]),
    )


def nll(labels, probabilities):
    p = np.clip(probabilities, 1e-12, 1 - 1e-12)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log1p(-p)))


def calibration_bins(labels, probabilities, bins=10):
    indices = np.minimum((probabilities * bins).astype(int), bins - 1)
    points = []
    for index in range(bins):
        mask = indices == index
        if mask.any():
            points.append((float(probabilities[mask].mean()),
                           float(labels[mask].mean()), int(mask.sum())))
    return points


def metrics(labels, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0,
    )
    points = calibration_bins(labels, probabilities)
    return {
        "examples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1.mean()),
        "auroc": float(roc_auc_score(labels, probabilities))
                 if len(np.unique(labels)) == 2 else None,
        "threshold": float(threshold),
        "precision_nc": float(precision[0]), "recall_nc": float(recall[0]),
        "f1_nc": float(f1[0]), "support_nc": int(support[0]),
        "precision_ad": float(precision[1]), "recall_ad": float(recall[1]),
        "f1_ad": float(f1[1]), "support_ad": int(support[1]),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "nll": nll(labels, probabilities),
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
        "ece_ad_probability": float(sum(abs(p - y) * count for p, y, count in points)
                                    / len(labels)),
    }


def fit_temperature(rows, margins):
    # Optimize the true patient-level objective; correlated slices do not each
    # get an independent vote in the calibration loss.
    def objective(log_temperature):
        _, labels, p = group_predictions(rows, expit(margins / np.exp(log_temperature)), "patient")
        return nll(labels, p)
    result = minimize_scalar(objective, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Temperature fitting failed")
    # Keep T=1 if optimization does not improve on the uncalibrated model.
    return float(np.exp(result.x)) if result.fun < objective(0.0) else 1.0


def choose_threshold(labels, probabilities):
    # Fixed grid, including 0.5. Deterministic ties favor the value nearest 0.5.
    candidates = np.linspace(0.0, 1.0, 201)
    scores = [metrics(labels, probabilities, float(t))["macro_f1"] for t in candidates]
    best = max(scores)
    tied = [float(t) for t, score in zip(candidates, scores) if abs(score - best) < 1e-12]
    return min(tied, key=lambda t: (abs(t - 0.5), t))


def selective_result(labels, probabilities, threshold, reject_threshold):
    predictions = (probabilities >= threshold).astype(int)
    confidence = np.where(predictions == 1, probabilities, 1 - probabilities)
    accepted = confidence >= reject_threshold
    count = int(accepted.sum())
    return {
        "reject_threshold": float(reject_threshold), "accepted": count,
        "rejected": int(len(labels) - count), "coverage": float(accepted.mean()),
        "accuracy_on_accepted": float((predictions[accepted] == labels[accepted]).mean())
                                if count else None,
        "accepted_nc": int(((labels == 0) & accepted).sum()),
        "accepted_ad": int(((labels == 1) & accepted).sum()),
    }


def choose_reject_threshold(labels, p, threshold, target, minimum_coverage):
    candidates = [selective_result(labels, p, threshold, float(t))
                  for t in np.r_[0.0, np.linspace(0.5, 0.99, 50)]]
    feasible = [r for r in candidates if r["coverage"] >= minimum_coverage
                and r["accuracy_on_accepted"] is not None
                and r["accuracy_on_accepted"] >= target]
    if not feasible:
        return {"target_met_on_validation": False,
                "reason": "No candidate meets the accuracy and coverage targets; rejection disabled.",
                **candidates[0]}
    best = max(feasible, key=lambda r: (r["coverage"], -r["reject_threshold"]))
    return {"target_met_on_validation": True, **best}


def predict(model, dataset, batch_size, workers, device, amp):
    import torch
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.type == "cuda")
    rows, margins = [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError("Expected two class logits per image")
            difference = (logits.float()[:, 1] - logits.float()[:, 0]).cpu().numpy()
            if not np.isfinite(difference).all():
                raise ValueError("Non-finite model logits")
            margins.extend(difference.tolist())
            for i in range(len(difference)):
                rows.append({
                    "label": int(batch["label"][i]),
                    "scan_id": batch["scan_id"][i],
                    "patient_id": batch["patient_id"][i],
                    "slice_index": int(batch["slice_index"][i]),
                    "path": batch["path"][i],
                })
    if len(rows) != len(dataset):
        raise RuntimeError("Incomplete evaluation")
    return rows, np.array(margins, dtype=np.float64)


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_predictions(out, split, rows, margins, settings):
    raw = expit(margins)
    calibrated = expit(margins / settings["temperature"])
    report = {"split": split, "temperature": settings["temperature"], "levels": {}}
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for column, level in enumerate(LEVELS):
        ids, labels, raw_p = group_predictions(rows, raw, level)
        _, _, p = group_predictions(rows, calibrated, level)
        threshold = settings["thresholds"][level]
        values = {
            "raw_0.5": metrics(labels, raw_p),
            "temperature_0.5": metrics(labels, p),
            "temperature_tuned": metrics(labels, p, threshold),
        }
        report["levels"][level] = values
        save_csv(out / f"{split}_{level}_predictions.csv", [
            {"id": identifier, "label": int(y), "raw_ad_probability": float(rp),
             "calibrated_ad_probability": float(cp), "threshold": threshold,
             "prediction": int(cp >= threshold)}
            for identifier, y, rp, cp in zip(ids, labels, raw_p, p)
        ])
        ax = axes[0, column]
        ax.plot([0, 1], [0, 1], "--", color="gray")
        for name, probabilities in (("Raw", raw_p), ("Temperature", p)):
            points = calibration_bins(labels, probabilities)
            ax.plot([b[0] for b in points], [b[1] for b in points], "o-", label=name)
        ax.set(xlim=(0, 1), ylim=(0, 1), title=f"{level}: reliability",
               xlabel="Mean predicted P(AD)", ylabel="Observed AD fraction")
        ax.legend()
        cm = np.array(values["temperature_tuned"]["confusion_matrix"])
        ax = axes[1, column]
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["NC", "AD"],
               yticklabels=["NC", "AD"], xlabel="Predicted", ylabel="True",
               title=f"{level}: calibrated, tuned threshold")
        print(f"{split} {level}:", flush=True)
        for method in METHODS:
            m = values[method]
            print(f"  {method:20} accuracy={m['accuracy']:.4f} "
                  f"macro-F1={m['macro_f1']:.4f} AUROC={m['auroc']} "
                  f"AD recall={m['recall_ad']:.4f}", flush=True)
    fig.tight_layout()
    fig.savefig(out / f"{split}_reliability_confusion.png", dpi=160)
    plt.close(fig)
    ids, labels, p = group_predictions(rows, calibrated, "patient")
    threshold = settings["thresholds"]["patient"]
    reject = settings["patient_rejection"]["reject_threshold"]
    report["patient_selective"] = selective_result(labels, p, threshold, reject)
    curve = [selective_result(labels, p, threshold, float(t))
             for t in np.r_[0.0, np.linspace(0.5, 0.99, 50)]]
    save_csv(out / f"{split}_patient_coverage.csv", curve)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    _, _, raw_patient = group_predictions(rows, raw, "patient")
    for name, probabilities in (("Raw", raw_patient), ("Temperature", p)):
        if len(np.unique(labels)) == 2:
            fpr, tpr, _ = roc_curve(labels, probabilities)
            axes[0].plot(fpr, tpr, label=name)
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Patient ROC")
    axes[0].legend()
    valid = [r for r in curve if r["accuracy_on_accepted"] is not None]
    axes[1].plot([r["coverage"] for r in valid],
                 [1 - r["accuracy_on_accepted"] for r in valid], "o-")
    axes[1].set(xlabel="Coverage (fraction accepted)", ylabel="Error rate among accepted",
                title="Patient risk versus coverage", xlim=(0, 1), ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(out / f"{split}_patient_roc_coverage.png", dpi=160)
    plt.close(fig)
    save_json(out / f"{split}_metrics.json", report)
    enriched = [dict(r, logit_margin=float(m), raw_ad_probability=float(rp),
                     calibrated_ad_probability=float(cp))
                for r, m, rp, cp in zip(rows, margins, raw, calibrated)]
    save_csv(out / f"{split}_slice_details.csv", enriched)
    failure_cases(out, split, rows, calibrated, ids, labels, p, threshold)
    return report


def failure_cases(out, split, rows, slice_p, ids, labels, patient_p, threshold):
    from PIL import Image
    cases = []
    for pid, label, p in zip(ids, labels, patient_p):
        pred = int(p >= threshold)
        if pred != label:
            cases.append({"patient_id": pid, "label": int(label), "prediction": pred,
                          "ad_probability": float(p),
                          "assigned_class_probability": float(p if pred else 1 - p)})
    cases.sort(key=lambda r: (-r["assigned_class_probability"], r["patient_id"]))
    selected = cases[:5]
    # Always create JSON, even when the model makes no errors.
    save_json(out / f"{split}_failure_cases.json", selected)
    if not selected:
        return
    fig, axes = plt.subplots(1, len(selected), figsize=(3 * len(selected), 3.6), squeeze=False)
    for ax, case in zip(axes[0], selected):
        indices = [i for i, r in enumerate(rows) if r["patient_id"] == case["patient_id"]]
        index = min(indices, key=lambda i: abs(slice_p[i] - case["ad_probability"]))
        case["representative_slice"] = rows[index]["path"]
        with Image.open(rows[index]["path"]) as image:
            ax.imshow(np.array(image.convert("L")), cmap="gray")
        actual, predicted = ("AD" if case[k] else "NC" for k in ("label", "prediction"))
        ax.set_title(f"{case['patient_id']}\nTrue {actual}; predicted {predicted}\n"
                     f"Patient P(AD)={case['ad_probability']:.3f}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Representative slices from up to five confident patient errors")
    fig.tight_layout()
    fig.savefig(out / f"{split}_failure_cases.png", dpi=160)
    plt.close(fig)
    save_json(out / f"{split}_failure_cases.json", selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["calibrate", "test"], default="calibrate")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--metadata-path", type=Path)
    parser.add_argument("--reject-target-accuracy", type=float, default=0.90)
    parser.add_argument("--minimum-coverage", type=float, default=0.50)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid batch size or worker count")
    if not 0 < args.reject_target_accuracy <= 1 or not 0 < args.minimum_coverage <= 1:
        parser.error("Accuracy and coverage targets must be in (0, 1]")
    settings_path = args.output_dir / "calibration.json"
    if args.mode == "calibrate" and settings_path.exists():
        parser.error("Calibration already exists. Use a new output directory to preserve this run.")
    if args.mode == "test" and not settings_path.exists():
        parser.error("Run --mode calibrate first in this output directory")

    # Lazy imports let numerical aggregation/calibration be tested without GPUs.
    from dataset import create_datasets
    from train import create_model, load_checkpoint, select_device, set_reproducibility
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    saved = checkpoint["arguments"]
    seed, image_size = int(saved["seed"]), int(saved["image_size"])
    set_reproducibility(seed)
    data_root = args.data_root or saved["data_root"]
    metadata = args.metadata_path or saved["metadata_path"]
    datasets = create_datasets(data_root=data_root, metadata_path=metadata,
                               seed=seed, image_size=image_size, output_channels=1)
    identity = {"checkpoint_sha256": checkpoint_hash(args.checkpoint),
                "model": checkpoint["model_name"], "epoch": int(checkpoint["epoch"]),
                "seed": seed, "image_size": image_size,
                "splits": split_manifest(datasets)}
    amp = device.type == "cuda" and not saved.get("no_amp", False)
    identity["mixed_precision"] = amp
    if args.mode == "test":
        settings = json.loads(settings_path.read_text())
        if identity != settings["identity"]:
            raise RuntimeError("Checkpoint, split or precision changed since calibration")
        if (args.output_dir / "test_metrics.json").exists():
            raise RuntimeError("Test results already exist; do not overwrite the final evaluation")
    model = create_model(checkpoint["model_name"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = "validation" if args.mode == "calibrate" else "test"
    print(f"Model={identity['model']} best epoch={identity['epoch']} split={split}", flush=True)
    rows, margins = predict(model, datasets[split], args.batch_size, args.workers, device, amp)
    if args.mode == "calibrate":
        temperature = fit_temperature(rows, margins)
        calibrated = expit(margins / temperature)
        thresholds = {}
        for level in LEVELS:
            _, labels, p = group_predictions(rows, calibrated, level)
            thresholds[level] = choose_threshold(labels, p)
        _, labels, p = group_predictions(rows, calibrated, "patient")
        rejection = choose_reject_threshold(labels, p, thresholds["patient"],
                                            args.reject_target_accuracy, args.minimum_coverage)
        settings = {"schema_version": 1, "identity": identity, "temperature": temperature,
                    "temperature_objective": "patient negative log likelihood",
                    "thresholds": thresholds, "threshold_objective": "validation macro-F1",
                    "threshold_grid_step": 0.005, "patient_rejection": rejection,
                    "reject_target_accuracy": args.reject_target_accuracy,
                    "minimum_coverage": args.minimum_coverage,
                    "checkpoint_validation_results": checkpoint.get("validation_results"),
                    "note": "Validation was also used for checkpoint selection; development scores are optimistic."}
        save_json(settings_path, settings)
        print(f"Frozen temperature={temperature:.4f}; thresholds={thresholds}", flush=True)
        print(f"Validation rejection selection: {rejection}", flush=True)
    report_predictions(args.output_dir, split, rows, margins, settings)
    print(f"Results saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
