"""Camera-realistic corruption of the policy's input image, for stress tests and augmentation.

jar_scene randomises the RENDER (light direction/intensity, robot and object colours) and the
BACKGROUND photo (brightness, tint, shift, a 3x3 blur on half the episodes). What it never
touches is the image as a whole, the way a real camera does: exposure, white balance, defocus
and motion blur, sensor noise, the tone curve, compression. Those act on the arm and the object
as much as on the room, so background diversity says nothing about them.

OBS_CORRUPT is a comma-separated spec; every op is optional:

    exposure=0.7     linear gain (0.7 = -0.5 EV, 1.4 = +0.5 EV); highlights clip, as they do
    wb=warm|cool|green   white-balance error (tungsten / shade / fluorescent cast)
    blur=1.5         defocus, Gaussian sigma in pixels of the policy image
    mblur=7          motion blur, kernel length in pixels, one random direction per process
    noise=0.03       sensor noise: shot noise ~ sqrt(signal) plus a read-noise floor
    vignette=0.4     radial falloff, fraction of light lost in the corners
    gamma=0.75       tone-curve mismatch between cameras (<1 lifts shadows)
    contrast=0.8     contrast about mid-grey
    jpeg=40          JPEG quality (Continuity Camera and most webcams deliver compressed video)

Ops are applied in the order a camera applies them, regardless of the order in the spec:
blur -> exposure/vignette -> noise (all in linear light) -> white balance -> tone curve ->
compression. Example: OBS_CORRUPT="exposure=0.8,wb=warm,blur=1,noise=0.02,jpeg=50".
"""
import os

import cv2
import numpy as np

WB = {"warm": (1.18, 1.00, 0.80), "cool": (0.85, 1.00, 1.18), "green": (0.95, 1.08, 0.95)}
_rng = np.random.default_rng(int(os.environ.get("OBS_CORRUPT_SEED", "1234")) + os.getpid())
_mblur_angle = float(_rng.uniform(0, np.pi))


def parse(spec):
    ops = {}
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        k, _, v = tok.partition("=")
        ops[k.strip()] = v.strip()
    unknown = set(ops) - {"exposure", "wb", "blur", "mblur", "noise", "vignette", "gamma", "contrast", "jpeg"}
    if unknown:
        raise ValueError(f"OBS_CORRUPT: unknown op(s) {sorted(unknown)}")
    return ops


def _motion_kernel(length, angle):
    k = max(3, int(round(length)) | 1)
    ker = np.zeros((k, k), np.float32)
    c = k // 2
    dx, dy = np.cos(angle) * c, np.sin(angle) * c
    cv2.line(ker, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
    return ker / max(ker.sum(), 1e-6)


def corrupt(img, ops):
    """img: HxWx3 uint8 RGB. Returns the same shape and dtype."""
    if not ops:
        return img
    x = (img.astype(np.float32) / 255.0) ** 2.2                       # to linear light
    if "blur" in ops:
        x = cv2.GaussianBlur(x, (0, 0), float(ops["blur"]))
    if "mblur" in ops:
        x = cv2.filter2D(x, -1, _motion_kernel(float(ops["mblur"]), _mblur_angle))
    if "exposure" in ops:
        x = x * float(ops["exposure"])
    if "vignette" in ops:
        h, w = x.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        x = x * (1.0 - float(ops["vignette"]) * np.clip(r2 / 2.0, 0, 1))[..., None]
    if "noise" in ops:
        s = float(ops["noise"])
        x = x + _rng.normal(0, 1, x.shape).astype(np.float32) * (s * np.sqrt(np.clip(x, 0, None)) + 0.3 * s * s ** 0.5)
    if "wb" in ops:
        x = x * np.float32(WB[ops["wb"]])
    x = np.clip(x, 0.0, 1.0) ** (1.0 / 2.2)                           # back to display space
    if "gamma" in ops:
        x = x ** float(ops["gamma"])
    if "contrast" in ops:
        x = (x - 0.5) * float(ops["contrast"]) + 0.5
    out = (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if "jpeg" in ops:
        ok, buf = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(float(ops["jpeg"]))])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].copy()
    return out


_ENV_OPS = None


def from_env(img):
    """Apply whatever OBS_CORRUPT asks for (parsed once per process)."""
    global _ENV_OPS
    if _ENV_OPS is None:
        _ENV_OPS = parse(os.environ.get("OBS_CORRUPT", ""))
    return corrupt(img, _ENV_OPS)
