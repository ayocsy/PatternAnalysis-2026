"""PyTorch models for AD versus normal-control MRI classification.

The module provides a small CNN baseline and a ConvNeXt-Tiny classifier. The
ConvNeXt design follows the architecture described by Liu et al. in "A ConvNet
for the 2020s" (2022): https://arxiv.org/abs/2201.03545.

This implementation is written specifically for the COMP3710 project and uses
only PyTorch model components.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch
from torch import Tensor, nn


class BaselineCNN(nn.Module):
    """A conventional CNN used as the low-complexity comparison model."""

    def __init__(self, input_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__()

        self.features = nn.Sequential(
            self._convolution_block(input_channels, 32),
            self._convolution_block(32, 64),
            self._convolution_block(64, 128),
            self._convolution_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.30),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _convolution_block(input_channels: int, output_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, images: Tensor) -> Tensor:
        features = self.features(images)
        pooled_features = self.pool(features)
        return self.classifier(pooled_features)


class LayerNorm2d(nn.Module):
    """Apply LayerNorm across channels while accepting NCHW image tensors."""

    def __init__(self, channels: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels, eps=epsilon)

    def forward(self, inputs: Tensor) -> Tensor:
        # LayerNorm expects the normalized channel dimension to be last.
        inputs = inputs.permute(0, 2, 3, 1)
        inputs = self.normalization(inputs)
        return inputs.permute(0, 3, 1, 2)


class DropPath(nn.Module):
    """Randomly drop entire residual branches during training.

    Unlike ordinary dropout, every spatial position and channel in one sample
    shares the same random mask. Evaluation mode always returns the input.
    """

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("Drop-path probability must be in [0, 1)")
        self.probability = probability

    def forward(self, inputs: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return inputs

        keep_probability = 1.0 - self.probability
        mask_shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        random_mask = torch.empty(
            mask_shape,
            dtype=inputs.dtype,
            device=inputs.device,
        ).bernoulli_(keep_probability)

        return inputs * random_mask / keep_probability


class ConvNeXtBlock(nn.Module):
    """The main ConvNeXt residual block.

    Processing order:
    7x7 depthwise convolution -> LayerNorm -> 4x channel expansion -> GELU ->
    channel reduction -> layer scaling -> stochastic residual connection.
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 4,
        layer_scale_initial_value: float = 1e-6,
        drop_path_probability: float = 0.0,
    ) -> None:
        super().__init__()

        hidden_channels = channels * expansion
        self.depthwise_convolution = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
        )
        self.normalization = nn.LayerNorm(channels, eps=1e-6)
        self.expand = nn.Linear(channels, hidden_channels)
        self.activation = nn.GELU()
        self.reduce = nn.Linear(hidden_channels, channels)
        self.layer_scale = nn.Parameter(
            layer_scale_initial_value * torch.ones(channels)
        )
        self.drop_path = DropPath(drop_path_probability)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs

        outputs = self.depthwise_convolution(inputs)
        outputs = outputs.permute(0, 2, 3, 1)
        outputs = self.normalization(outputs)
        outputs = self.expand(outputs)
        outputs = self.activation(outputs)
        outputs = self.reduce(outputs)
        outputs = outputs * self.layer_scale
        outputs = outputs.permute(0, 3, 1, 2)

        return residual + self.drop_path(outputs)


class ConvNeXt(nn.Module):
    """Configurable ConvNeXt image classifier for single-channel MRI slices."""

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        depths: Sequence[int] = (3, 3, 9, 3),
        dimensions: Sequence[int] = (96, 192, 384, 768),
        stochastic_depth_probability: float = 0.10,
        layer_scale_initial_value: float = 1e-6,
    ) -> None:
        super().__init__()

        if len(depths) != 4 or len(dimensions) != 4:
            raise ValueError("ConvNeXt requires four stage depths and dimensions")
        if any(depth <= 0 for depth in depths):
            raise ValueError("Every ConvNeXt stage must contain at least one block")

        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                dimensions[0],
                kernel_size=4,
                stride=4,
            ),
            LayerNorm2d(dimensions[0]),
        )

        total_blocks = sum(depths)
        drop_probabilities = torch.linspace(
            0.0,
            stochastic_depth_probability,
            total_blocks,
        ).tolist()

        stages: list[nn.Module] = []
        downsample_layers: list[nn.Module] = []
        block_index = 0

        for stage_index, (depth, channels) in enumerate(zip(depths, dimensions)):
            blocks = []
            for _ in range(depth):
                blocks.append(
                    ConvNeXtBlock(
                        channels=channels,
                        layer_scale_initial_value=layer_scale_initial_value,
                        drop_path_probability=drop_probabilities[block_index],
                    )
                )
                block_index += 1

            stages.append(nn.Sequential(*blocks))

            if stage_index < len(dimensions) - 1:
                downsample_layers.append(
                    nn.Sequential(
                        LayerNorm2d(channels),
                        nn.Conv2d(
                            channels,
                            dimensions[stage_index + 1],
                            kernel_size=2,
                            stride=2,
                        ),
                    )
                )

        self.stages = nn.ModuleList(stages)
        self.downsample_layers = nn.ModuleList(downsample_layers)
        self.final_normalization = nn.LayerNorm(dimensions[-1], eps=1e-6)
        self.classifier = nn.Linear(dimensions[-1], num_classes)

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, images: Tensor) -> Tensor:
        features = self.stem(images)

        for stage_index, stage in enumerate(self.stages):
            features = stage(features)
            if stage_index < len(self.downsample_layers):
                features = self.downsample_layers[stage_index](features)

        # Global average pooling converts BxCxHxW into BxC.
        pooled_features = features.mean(dim=(-2, -1))
        return self.final_normalization(pooled_features)

    def forward(self, images: Tensor) -> Tensor:
        features = self.forward_features(images)
        return self.classifier(features)


def convnext_tiny(input_channels: int = 1, num_classes: int = 2) -> ConvNeXt:
    """Construct the standard ConvNeXt-Tiny configuration."""

    return ConvNeXt(
        input_channels=input_channels,
        num_classes=num_classes,
        depths=(3, 3, 9, 3),
        dimensions=(96, 192, 384, 768),
        stochastic_depth_probability=0.10,
    )


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of parameters updated by the optimizer."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def smoke_test(batch_size: int = 2, image_size: int = 224) -> None:
    """Verify that both models accept MRI batches and produce two logits."""

    images = torch.randn(batch_size, 1, image_size, image_size)
    models = {
        "Baseline CNN": BaselineCNN(),
        "ConvNeXt-Tiny": convnext_tiny(),
    }

    print("Model smoke test")
    print("-" * 68)

    with torch.no_grad():
        for model_name, model in models.items():
            model.eval()
            logits = model(images)
            expected_shape = (batch_size, 2)

            if tuple(logits.shape) != expected_shape:
                raise RuntimeError(
                    f"{model_name} returned {tuple(logits.shape)}, "
                    f"expected {expected_shape}"
                )

            parameter_count = count_trainable_parameters(model)
            print(
                f"{model_name:15} output={tuple(logits.shape)}  "
                f"parameters={parameter_count:,}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the project models")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    smoke_test(arguments.batch_size, arguments.image_size)
