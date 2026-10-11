"""Transfer official ImageNet weights into the project's custom ConvNeXt.

Architecture: Liu et al., A ConvNet for the 2020s (CVPR 2022),
https://arxiv.org/abs/2201.03545
Weight names and pretrained checkpoint follow the official torchvision source:
https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
No torchvision model replaces the project's modules.py implementation.
"""
from __future__ import annotations


def source_key(destination: str) -> str:
    """Map a custom backbone state key to its official torchvision equivalent."""
    if destination.startswith('stem.0.'):
        return destination.replace('stem.0.', 'features.0.0.', 1)
    if destination.startswith('stem.1.normalization.'):
        return destination.replace('stem.1.normalization.', 'features.0.1.', 1)
    if destination.startswith('stages.'):
        parts = destination.split('.')
        stage, block, component = int(parts[1]), parts[2], parts[3]
        prefix = f'features.{2 * stage + 1}.{block}.'
        if component == 'layer_scale':
            return prefix + 'layer_scale'
        indices = {'depthwise_convolution': 0, 'normalization': 2,
                   'expand': 3, 'reduce': 5}
        return prefix + f'block.{indices[component]}.' + '.'.join(parts[4:])
    if destination.startswith('downsample_layers.'):
        parts = destination.split('.')
        prefix = f'features.{2 * int(parts[1]) + 2}.{parts[2]}.'
        tail = parts[3:]
        if tail[0] == 'normalization':
            tail = tail[1:]
        return prefix + '.'.join(tail)
    if destination.startswith('final_normalization.'):
        return destination.replace('final_normalization.', 'classifier.0.', 1)
    raise ValueError(f'Unsupported backbone key: {destination}')


def initialize_pretrained(model):
    """Copy every backbone tensor, then verify numerical feature equivalence.

Summing the RGB stem kernels is exactly equivalent to repeating the normalized
single-channel input into three channels. We keep the existing MRI preprocessing;
we do not claim to use the official RGB ImageNet inference transforms.
"""
    import torch
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    source = convnext_tiny(weights=weights).cpu().eval()
    official = source.state_dict()
    current = model.state_dict()
    consumed = set()
    for key, tensor in current.items():
        if key.startswith('classifier.'):
            continue  # Keep the project's freshly initialized two-class head.
        original_key = source_key(key)
        value = official[original_key]
        consumed.add(original_key)
        if key == 'stem.0.weight':
            value = value.sum(dim=1, keepdim=True)
        elif key.endswith('.layer_scale'):
            value = value.reshape(-1)
        if tensor.shape != value.shape:
            raise RuntimeError(f'Weight shape mismatch for {key}: {tensor.shape} vs {value.shape}')
        current[key] = value
    expected = {key for key in official if not key.startswith('classifier.2.')}
    if consumed != expected:
        raise RuntimeError(f'Incomplete backbone transfer: missing={expected-consumed}, extra={consumed-expected}')
    model.load_state_dict(current, strict=True)
    model.cpu().eval()
    generator = torch.Generator().manual_seed(3710)
    image = torch.randn(1, 1, 64, 64, generator=generator)
    with torch.no_grad():
        reference = source.classifier[0](source.features(image.repeat(1, 3, 1, 1)).mean((-2, -1), keepdim=True)).flatten(1)
        actual = model.forward_features(image)
        torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)
    error = float((actual - reference).abs().max())
    print(f'Pretrained backbone transfer verified: {len(consumed)} tensors; max feature error={error:.3g}', flush=True)
    return {'weights': 'ConvNeXt_Tiny_Weights.IMAGENET1K_V1',
            'url': weights.url, 'backbone_tensors': len(consumed),
            'max_feature_error': error, 'stem_adaptation': 'sum RGB kernels',
            'preprocessing': 'existing MRI grayscale resize/pad and mean=std=0.5'}


def download_weights():
    """Cache official weights without GPU training; safe to run on login node."""
    from torchvision.models import ConvNeXt_Tiny_Weights
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    weights.get_state_dict(progress=True, check_hash=True)
    print(f'Cached official weights: {weights.url}', flush=True)
