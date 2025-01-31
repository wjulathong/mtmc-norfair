import warnings
from pathlib import Path

import torch
import torchvision.transforms.v2 as T
from cv2.typing import MatLike

from ..models import build_model
from .model_complexity import compute_model_complexity
from .torchtools import load_pretrained_weights


class FeatureExtractor:
    def __init__(
        self,
        model_name: str | None = None,
        model_path: Path | None = None,
        image_size: tuple[int, int] = (256, 128),
        pixel_mean: list[float] = [0.485, 0.456, 0.406],
        pixel_std: list[float] = [0.229, 0.224, 0.225],
        pixel_norm: bool = True,
        device: str = "cuda",
        verbose: bool = True,
    ) -> None:
        if model_path is not None:
            if not model_path.is_file():
                warnings.warn(f'No file found at "{model_path.resolve()}"')

        # Build model
        model = build_model(
            model_name if model_name is not None else "",
            num_classes=1,
            pretrained=not (model_path is not None and model_path.is_file()),
            use_gpu=device.startswith("cuda"),
        )
        model.to(device)
        model.eval()

        if verbose:
            num_params, flops = compute_model_complexity(model, (1, 3, *image_size))
            print(
                f"Model: {model_name if model_name is not None else ''}\n"
                f"- params: {num_params:,}\n"
                f"- flops: {flops:,}"
            )

        if model_path is not None and model_path.is_file():
            load_pretrained_weights(model, model_path)

        # Build transforms functions
        transforms = [
            T.ToImage(),
            T.Resize(image_size),
            T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
        ]
        if pixel_norm:
            transforms.append(T.Normalize(mean=pixel_mean, std=pixel_std))
        preprocess = T.Compose(transforms)

        # Attributes
        self.model = model
        self.preprocess = preprocess
        self.device = device

    def __call__(self, image: MatLike) -> torch.Tensor:
        images = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(images)

        return features.cpu()
