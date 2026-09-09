"""Device-aware inference for the original 44-channel CubiCasa checkpoint."""
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn.functional import interpolate

from floortrans.models import get_model
from floortrans.loaders.augmentations import RotateNTurns

SPLIT = (21, 12, 11)
ROOM_CLASSES = ("Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bedroom",
                "Bath", "Hallway", "Railing", "Storage", "Garage", "Other rooms")
ICON_CLASSES = ("Empty", "Window", "Door", "Closet", "Electr. Appl.", "Toilet",
                "Sink", "Sauna bench", "Fire Place", "Bathtub", "Chimney")


def resolve_device(device="auto"):
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are auto, cpu, cuda, and cuda:N.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Use --device cpu or install a CUDA build on a GPU host.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device.index} does not exist.")
    return device


def load_model(checkpoint, device="auto", arch="hg_furukawa_original"):
    device = resolve_device(device)
    # Load on CPU first, including checkpoints originally saved from CUDA.
    # Never fall back to unrestricted pickle loading.
    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(saved, Mapping):
        raise ValueError("Expected a state dictionary or a checkpoint containing model_state.")
    state = saved.get("model_state", saved)
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint model_state must be a state dictionary.")
    # DataParallel adds this prefix; only strip it when all keys have it.
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model = get_model(arch, sum(SPLIT), pretrained=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def prepare_image(path, max_size=512):
    """RGB, aspect-preserving resize, [-1, 1] normalization, no square padding.

    Output coordinates refer to the returned resized image. max_size=0 retains
    the original resolution. The four hourglass pooling levels need >=64 px.
    """
    if max_size != 0 and max_size < 64:
        raise ValueError("max_size must be 0 (original size) or at least 64.")
    with Image.open(path) as source:
        original = ImageOps.exif_transpose(source).convert("RGBA")
        original_size = original.size
        image = Image.new("RGBA", original.size, "white")
        image.alpha_composite(original)
        image = image.convert("RGB")
    if max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
    if min(image.size) < 64:
        raise ValueError("Both processed image dimensions must be at least 64 pixels; increase --max-size for narrow plans.")
    pixels = np.array(image, dtype=np.float32)
    tensor = torch.from_numpy((pixels / 127.5 - 1).transpose(2, 0, 1).copy()).unsqueeze(0)
    return image, tensor, original_size


@torch.inference_mode()
def predict(model, images, size=None, rotate=True):
    """Average logits/heatmaps across the original four rotations on model device."""
    if images.ndim != 4 or images.shape[0] != 1 or images.shape[1] != 3:
        raise ValueError("Expected a single RGB image tensor [1, 3, H, W].")
    if min(images.shape[-2:]) < 64:
        raise ValueError("Input height and width must be at least 64 pixels.")
    images = images.to(next(model.parameters()).device)
    size = size or images.shape[-2:]
    rot = RotateNTurns()
    rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)] if rotate else [(0, 0)]
    prediction = None
    for forward, back in rotations:
        pred = model(rot(images, "tensor", forward))
        pred = rot(rot(pred, "tensor", back), "points", back)
        pred = interpolate(pred, size=size, mode="bilinear", align_corners=True)
        prediction = pred if prediction is None else prediction + pred
    return prediction / len(rotations)
