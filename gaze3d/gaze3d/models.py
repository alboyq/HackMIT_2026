"""Wrappers around pretrained appearance-based gaze networks.

Every wrapper takes a batch of ETH-XGaze-normalized 224x224 BGR uint8 face crops and
returns unit gaze vectors in the *normalized* camera frame (XGaze convention:
x = -cos(p) sin(y), y = -sin(p), z = -cos(p) cos(y)). Horizontal-flip test-time
augmentation is applied by default: the flipped image's yaw is negated and the two
directions averaged, which removes a good part of each net's left/right bias.
"""
from __future__ import annotations
import os, sys, time, importlib
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from .normalize import pitchyaw_to_vec

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEIGHTS = os.path.join(ROOT, "models", "weights")
THIRD = os.path.join(HERE, "third_party")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class GazeModel:
    """Base: subclasses set self.net, self.rgb, self.imagenet, self.yaw_first."""
    name = "base"
    rgb = True
    imagenet = True
    yaw_first = False

    def __init__(self, device: torch.device):
        self.device = device
        self.net: nn.Module | None = None
        self.mean = IMAGENET_MEAN.to(device)
        self.std = IMAGENET_STD.to(device)
        self.last_ms = 0.0

    def _prep(self, crops_bgr: np.ndarray, flip: bool) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(crops_bgr)).to(self.device)
        if flip:
            x = torch.flip(x, dims=[2])
        x = x.permute(0, 3, 1, 2).float() / 255.0
        if self.rgb:
            x = x[:, [2, 1, 0]]
        if self.imagenet:
            x = (x - self.mean) / self.std
        return x

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict(self, crops_bgr: np.ndarray, tta_flip: bool = True) -> np.ndarray:
        """crops_bgr: (N,224,224,3) uint8 -> (N,3) unit vectors in normalized frame."""
        t0 = time.perf_counter()
        n = crops_bgr.shape[0]
        x = self._prep(crops_bgr, False)
        if tta_flip:
            x = torch.cat([x, self._prep(crops_bgr, True)], 0)
        out = self._forward(x).float().cpu().numpy()
        if self.yaw_first:
            out = out[:, [1, 0]]
        if tta_flip:
            a, b = out[:n], out[n:]
            b = b.copy(); b[:, 1] *= -1        # mirrored image -> mirrored yaw
            v = pitchyaw_to_vec(a) + pitchyaw_to_vec(b)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
        else:
            v = pitchyaw_to_vec(out)
        self.last_ms = (time.perf_counter() - t0) * 1000
        return v


class UniGaze(GazeModel):
    """UniGaze (Qin et al., WACV 2025): MAE-pretrained ViT trained jointly on ETH-XGaze,
    MPIIFaceGaze, GazeCapture, EyeDiap and Gaze360. Best published cross-dataset numbers
    for a downloadable checkpoint."""
    rgb, imagenet, yaw_first = True, True, False

    def __init__(self, device, variant="unigaze_h14_joint"):
        super().__init__(device)
        import unigaze
        self.name = variant
        self.net = unigaze.load(variant, device="cpu").to(device).eval()

    def _forward(self, x):
        return self.net(x)["pred_gaze"]


class XGazeResNet18(GazeModel):
    """ETH-XGaze ResNet-18 trained by hysts (ptgaze). MIT weights, ~4.9 deg on XGaze."""
    name = "xgaze_resnet18"
    rgb, imagenet, yaw_first = True, True, False

    def __init__(self, device):
        super().__init__(device)
        import timm
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("hysts/ptgaze-eth-xgaze-resnet18", "model.safetensors")
        net = timm.create_model("resnet18", num_classes=2)
        net.load_state_dict(load_file(path))
        self.net = net.to(device).eval()


def _gazehub_state_dict(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


class PureGazeR50(GazeModel):
    """PureGaze (Cheng et al., AAAI 2022) ResNet-50 trained on ETH-XGaze. Most robust
    zero-shot model in the independent Gaze4HRI benchmark. GazeHub conventions: BGR,
    0..1 scaling. Output order verified empirically as (pitch, yaw)."""
    name = "puregaze_r50"
    rgb, imagenet, yaw_first = False, False, False   # mirror test: output is (pitch, yaw)

    def __init__(self, device, path=os.path.join(WEIGHTS, "puregaze_res50_xgaze.pt")):
        super().__init__(device)
        sys.path.insert(0, THIRD)
        modules = importlib.import_module("puregaze_modules")
        sys.path.pop(0)
        feat = modules.resnet50(pretrained=False)
        head = modules.ResGazeEs()
        sd = _gazehub_state_dict(path)
        feat.load_state_dict({k[len("feature."):]: v for k, v in sd.items() if k.startswith("feature.")})
        head.load_state_dict({k[len("gazeEs."):]: v for k, v in sd.items() if k.startswith("gazeEs.")})
        self.net = nn.Sequential(feat, head).to(device).eval()


class GazeTRHybrid(GazeModel):
    """GazeTR-Hybrid (Cheng & Lu, ICPR 2022) pretrained on ETH-XGaze. GazeHub conventions."""
    name = "gazetr_hybrid"
    rgb, imagenet, yaw_first = False, False, True

    def __init__(self, device, path=os.path.join(WEIGHTS, "gazetr_hybrid_xgaze.pt")):
        super().__init__(device)
        sys.path.insert(0, THIRD)
        import gazetr_resnet                       # upstream does `from resnet import resnet18`
        sys.modules["resnet"] = gazetr_resnet
        gazetr = importlib.import_module("gazetr_model")
        sys.path.pop(0)
        net = gazetr.Model()
        missing, unexpected = net.load_state_dict(_gazehub_state_dict(path), strict=False)
        if missing:
            raise RuntimeError(f"GazeTR missing keys: {missing[:5]}")
        self.net = net.to(device).eval()
        self._pos = torch.arange(0, 50, device=device)

    def _forward(self, x):
        net = self.net
        feature = net.base_model(x)
        b = feature.size(0)
        feature = feature.flatten(2).permute(2, 0, 1)
        cls = net.cls_token.repeat((1, b, 1))
        feature = torch.cat([cls, feature], 0)
        pos_feature = net.pos_embedding(self._pos)
        feature = net.encoder(feature, pos_feature).permute(1, 2, 0)[:, :, 0]
        return net.feed(feature)


REGISTRY = {
    "unigaze_h14_joint": lambda d: UniGaze(d, "unigaze_h14_joint"),
    "unigaze_l16_joint": lambda d: UniGaze(d, "unigaze_l16_joint"),
    "unigaze_b16_joint": lambda d: UniGaze(d, "unigaze_b16_joint"),
    "xgaze_resnet18": XGazeResNet18,
    "puregaze_r50": PureGazeR50,
    "gazetr_hybrid": GazeTRHybrid,
}


@dataclass
class EnsembleOutput:
    mean: np.ndarray            # (3,) unit vector, normalized frame
    per_model: dict             # name -> (3,)
    ms: dict                    # name -> inference ms


class Ensemble:
    def __init__(self, names: list[str], device: torch.device | None = None, tta_flip=True):
        self.device = device or pick_device()
        self.models: list[GazeModel] = []
        self.tta_flip = tta_flip
        self.weights: dict[str, float] = {}
        for n in names:
            try:
                t0 = time.perf_counter()
                m = REGISTRY[n](self.device)
                self.models.append(m)
                self.weights[m.name] = 1.0
                print(f"[gaze3d] loaded {n} in {time.perf_counter()-t0:.1f}s", flush=True)
            except Exception as e:  # keep going with whatever loads
                print(f"[gaze3d] could not load {n}: {e!r}", flush=True)
        if not self.models:
            raise RuntimeError("no gaze models loaded")
        self.warmup()

    def warmup(self):
        x = np.zeros((1, 224, 224, 3), np.uint8)
        for m in self.models:
            for _ in range(2):
                m.predict(x, self.tta_flip)

    @property
    def names(self):
        return [m.name for m in self.models]

    def predict(self, crop_bgr: np.ndarray) -> EnsembleOutput:
        x = crop_bgr[None]
        per, ms, acc = {}, {}, np.zeros(3)
        for m in self.models:
            v = m.predict(x, self.tta_flip)[0]
            per[m.name] = v
            ms[m.name] = m.last_ms
            acc += self.weights.get(m.name, 1.0) * v
        n = np.linalg.norm(acc)
        return EnsembleOutput(mean=acc / n if n > 0 else np.array([0, 0, -1.0]), per_model=per, ms=ms)
