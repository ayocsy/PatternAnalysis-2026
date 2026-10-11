"""Validation-selected fine-tuning experiment; never evaluates test images.

Keeps the custom modules.py model, split and preprocessing. Uses ImageNet
initialization, a head-only phase, differential learning rates, cosine decay,
light label smoothing and patient-uniform sampling. Pretraining and these
training changes are a combined experiment, not isolated causal ablations.
"""
from __future__ import annotations
import argparse
from collections import Counter
import math
from pathlib import Path
import time
from convnext_transfer import download_weights, initialize_pretrained


def patient_sample_weights(records):
    """Equal expected total weight per patient; retain the patient class prior."""
    counts = Counter(record.patient_id for record in records)
    return [1.0 / counts[record.patient_id] for record in records]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--head-epochs', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--head-learning-rate', type=float, default=2.5e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--label-smoothing', type=float, default=0.05)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=3710)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='cuda')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--max-train-batches', type=int)
    parser.add_argument('--max-eval-batches', type=int)
    parser.add_argument('--profile-batches', type=int, default=20)
    parser.add_argument('--data-root', type=Path, default=Path('/home/groups/comp3710/ADNI/AD_NC'))
    parser.add_argument('--metadata-path', type=Path, default=Path('/home/groups/comp3710/ADNI/meta_data_with_label.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/convnext_pretrained_seed3710'))
    args = parser.parse_args()
    if args.download_only:
        download_weights()
        return
    if args.epochs < 1 or not 0 <= args.head_epochs < args.epochs:
        parser.error('Require epochs > head-epochs >= 0')
    if min(args.batch_size, args.patience, args.profile_batches, args.log_interval) < 1 or args.workers < 0:
        parser.error('Invalid batch size, patience, profile count, log interval or workers')
    if min(args.learning_rate, args.head_learning_rate) <= 0 or args.weight_decay < 0 or not 0 <= args.label_smoothing < 1:
        parser.error('Invalid optimization settings')
    if any(value is not None and value < 1 for value in (args.max_train_batches, args.max_eval_batches)):
        parser.error('Batch caps must be positive')
    if args.output_dir.exists():
        parser.error('Output directory already exists; choose a fresh location')
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from dataset import create_datasets
    from train import (create_loaders, create_model, evaluate, load_checkpoint,
                       plot_history, print_evaluation, profile_inference,
                       save_checkpoint, seed_worker, select_device,
                       set_reproducibility, train_one_epoch, write_json)
    from modules import count_trainable_parameters
    from evaluate_models import split_manifest
    set_reproducibility(args.seed)
    device = select_device(args.device)
    use_amp = device.type == 'cuda' and not args.no_amp
    args.model = 'convnext'
    args.skip_test = True
    args.experiment = 'imagenet_patient_uniform_finetuning_v1'
    datasets = create_datasets(data_root=args.data_root, metadata_path=args.metadata_path,
                               seed=args.seed, image_size=args.image_size, output_channels=1)
    loaders = create_loaders(datasets, args.batch_size, args.workers, args.seed, device.type == 'cuda')
    weights = patient_sample_weights(datasets['train'].records)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True,
                                   generator=torch.Generator().manual_seed(args.seed))
    loaders['train'] = DataLoader(datasets['train'], batch_size=args.batch_size,
        sampler=sampler, num_workers=args.workers, pin_memory=device.type == 'cuda',
        persistent_workers=args.workers > 0, worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed))
    model = create_model('convnext')
    transfer = initialize_pretrained(model)  # Fail early if any tensor mapping is wrong.
    model.to(device)
    backbone = [p for name, p in model.named_parameters() if not name.startswith('classifier.')]
    head = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {'params': backbone, 'lr': args.learning_rate},
        {'params': head, 'lr': args.head_learning_rate},
    ], weight_decay=args.weight_decay)
    total_finetune = args.epochs - args.head_epochs
    def backbone_schedule(epoch):
        if epoch < args.head_epochs:
            return 0.0
        progress = (epoch - args.head_epochs) / max(1, total_finetune - 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    def head_schedule(epoch):
        return 1.0 if epoch < args.head_epochs else backbone_schedule(epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [backbone_schedule, head_schedule])
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    evaluation_criterion = nn.CrossEntropyLoss()  # Keep validation loss comparable.
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / 'configuration.json', vars(args))
    write_json(args.output_dir / 'transfer_verification.json', transfer)
    write_json(args.output_dir / 'split_manifest.json', split_manifest(datasets))
    parameter_count = count_trainable_parameters(model)
    print(f'Model=convnext pretrained; device={device}; parameters={parameter_count:,}; AMP={use_amp}', flush=True)
    print('No test image inference. Selection uses validation patient macro-F1.', flush=True)
    print('Patient-uniform sampling uses replacement; epoch sample count is not unique image coverage.', flush=True)
    best_score, stale, history = -math.inf, 0, []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        head_only = epoch <= args.head_epochs
        for parameter in backbone:
            parameter.requires_grad_(not head_only)
        learning_rates = [g['lr'] for g in optimizer.param_groups]
        print(f'\nEpoch {epoch:03d}/{args.epochs:03d}; head_only={head_only}; LR={learning_rates}', flush=True)
        training = train_one_epoch(model, loaders['train'], criterion, optimizer, scaler,
                                  device, use_amp, args.log_interval, args.max_train_batches)
        validation = evaluate(model, loaders['validation'], evaluation_criterion, device,
                              use_amp, args.max_eval_batches)
        print(f'Train loss={training["loss"]:.4f}; slice accuracy={training["accuracy"]:.4f}', flush=True)
        print_evaluation('Validation', validation)
        history.append({'epoch': epoch, 'learning_rate': learning_rates[0],
                        'head_learning_rate': learning_rates[1], 'head_only': head_only,
                        'train': training, 'validation': validation})
        scheduler.step()
        write_json(args.output_dir / 'history.json', history)
        plot_history(history, args.output_dir / 'training_curves.png')
        save_checkpoint(args.output_dir / 'last_model.pth', model, optimizer, scheduler,
                        epoch, 'convnext', validation, args)
        score = validation['patient']['macro_f1']
        if score > best_score:
            best_score, stale = score, 0
            save_checkpoint(args.output_dir / 'best_model.pth', model, optimizer, scheduler,
                            epoch, 'convnext', validation, args)
            print('Saved best validation patient macro-F1 checkpoint.', flush=True)
        elif not head_only:
            stale += 1
        if not head_only and stale >= args.patience:
            print('Early stopping.', flush=True)
            break
    training_seconds = time.perf_counter() - started
    peak_training = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else None
    checkpoint = load_checkpoint(args.output_dir / 'best_model.pth', device)
    model.load_state_dict(checkpoint['model_state_dict'])
    profile = profile_inference(model, loaders['validation'], device, use_amp, args.profile_batches)
    write_json(args.output_dir / 'resource_profile.json', {
        'model': 'convnext', 'experiment': args.experiment, 'device': str(device),
        'trainable_parameters': parameter_count, 'training_seconds': training_seconds,
        'peak_training_memory_mb': peak_training, 'profile_split': 'validation', **profile})
    print(f'Completed: best epoch={checkpoint["epoch"]}; validation patient macro-F1={best_score:.4f}', flush=True)
    print(f'Training seconds={training_seconds:.2f}; saved to {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
