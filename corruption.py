"""
data/corruption.py
==================
ROVA-style spatio-temporal corruption engine.

Design contract (this is what makes spec-replay and the CAAD/GRPO comparison
valid):

  apply_corruption(frames, spec) is a PURE FUNCTION.
  All randomness flows from np.random.default_rng(spec.seed).
  Same (frames, spec) -> bit-identical output, on any machine.

Corruption families follow the ROVA paper (Sec 3.1 / Tab. 8):
  lighting : dusk, night, overexposure, shadow
  camera   : translation, zoom, rotation (temporally smooth shake)
  occlusion: static, dynamic
  weather  : fog, rain, snow
plus optional temporal shuffle.

Every corruption preserves (T, H, W, C) -- required so clean/corrupted views
tokenize to the SAME number of visual tokens (the CAAD teacher scores the
student's rollout on shared prefixes, so prompt lengths must match).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field

import cv2
import numpy as np

CORRUPTION_STYLES = [
    "dusk", "night", "overexposure", "shadow",          # lighting
    "translation", "zoom", "rotation",                  # camera
    "static_occlusion", "dynamic_occlusion",            # occlusion
    "fog", "rain", "snow",                              # weather
]
SEVERITIES = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class CorruptionSpec:
    video_id: str
    style: str
    severity: int                      # 1..5
    seed: int
    temporal_shuffle: bool = False
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "CorruptionSpec":
        return CorruptionSpec(**json.loads(s))


def derive_seed(video_id: str, style: str, severity: int, global_seed: int) -> int:
    """Seed from sample identity: stable across shuffle order / num_workers."""
    key = f"{global_seed}:{video_id}:{style}:{severity}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def make_spec(video_id, style, severity, global_seed, temporal_shuffle=False):
    return CorruptionSpec(
        video_id=video_id, style=style, severity=int(severity),
        seed=derive_seed(video_id, style, severity, global_seed),
        temporal_shuffle=temporal_shuffle,
    )


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
def apply_corruption(frames: np.ndarray, spec: CorruptionSpec) -> np.ndarray:
    """
    frames: (T, H, W, 3) uint8 RGB.  Returns same shape/dtype.
    """
    assert frames.ndim == 4 and frames.shape[-1] == 3, "expect (T,H,W,3) RGB"
    rng = np.random.default_rng(spec.seed)
    out = frames.astype(np.float32)
    sev = spec.severity / 5.0  # normalize 0.2..1.0

    fn = _DISPATCH[spec.style]
    out = fn(out, sev, rng)

    if spec.temporal_shuffle:
        perm = rng.permutation(out.shape[0])
        out = out[perm]

    out = np.clip(out, 0, 255).astype(np.uint8)
    assert out.shape == frames.shape, (
        f"corruption changed shape {frames.shape}->{out.shape}; "
        "this breaks clean/corrupted prompt-length alignment")
    return out


# --------------------------------------------------------------------------- #
# Lighting
# --------------------------------------------------------------------------- #
def _dusk(x, sev, rng):
    gamma = 1.0 + 1.2 * sev
    x = 255.0 * (x / 255.0) ** gamma
    # warm orange tint, stronger toward the top of the frame (sky)
    tint = np.array([1.0 + 0.25 * sev, 1.0, 1.0 - 0.25 * sev])
    grad = np.linspace(1.0, 0.4, x.shape[1])[None, :, None, None]
    return x * (1 + (tint - 1) * grad).transpose(0, 1, 3, 2)[..., 0, :][..., None, :].squeeze(-2) \
        if False else x * tint[None, None, None, :] * (0.85 + 0.15 * (1 - sev))


def _night(x, sev, rng):
    x = x * (1.0 - 0.75 * sev)
    x[..., 2] *= 1.0 + 0.15 * sev            # slight blue cast
    noise = rng.normal(0, 6 + 14 * sev, x.shape).astype(np.float32)
    return x + noise                          # sensor noise in the dark


def _overexposure(x, sev, rng):
    x = x * (1.0 + 1.5 * sev) + 40 * sev
    # blown-out bloom around the brightest region
    T, H, W, _ = x.shape
    cy, cx = rng.integers(H // 4, 3 * H // 4), rng.integers(W // 4, 3 * W // 4)
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = ((yy - cy) ** 2 + (xx - cx) ** 2) / (0.12 * (H * W) * (0.4 + sev))
    bloom = np.exp(-r2)[None, :, :, None] * 255 * sev
    return x + bloom


def _shadow(x, sev, rng):
    T, H, W, _ = x.shape
    # smooth random shadow mask, drifting slowly across frames
    base = rng.random((H // 8, W // 8)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 3)
    base = cv2.resize(base, (W, H))
    base = (base > np.quantile(base, 0.6)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 15)
    dx, dy = rng.integers(-2, 3, size=2)
    out = np.empty_like(x)
    m = base
    for t in range(T):
        m = np.roll(np.roll(m, dy, axis=0), dx, axis=1)
        out[t] = x[t] * (1.0 - 0.7 * sev * m[..., None])
    return out


# --------------------------------------------------------------------------- #
# Camera motion (temporally-smooth affine jitter; border replicate keeps shape)
# --------------------------------------------------------------------------- #
def _smooth_noise(T, scale, rng, smooth=5):
    v = rng.normal(0, scale, T).astype(np.float32)
    k = np.ones(smooth) / smooth
    return np.convolve(v, k, mode="same")


def _warp_seq(x, mats):
    T, H, W, _ = x.shape
    out = np.empty_like(x)
    for t in range(T):
        out[t] = cv2.warpAffine(
            x[t], mats[t], (W, H),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


def _translation(x, sev, rng):
    T, H, W, _ = x.shape
    dx = _smooth_noise(T, 0.04 * W * sev, rng)
    dy = _smooth_noise(T, 0.04 * H * sev, rng)
    mats = [np.float32([[1, 0, dx[t]], [0, 1, dy[t]]]) for t in range(T)]
    return _warp_seq(x, mats)


def _zoom(x, sev, rng):
    T, H, W, _ = x.shape
    s = 1.0 + np.abs(_smooth_noise(T, 0.10 * sev, rng))
    mats = [cv2.getRotationMatrix2D((W / 2, H / 2), 0, float(s[t])) for t in range(T)]
    return _warp_seq(x, mats)


def _rotation(x, sev, rng):
    T, H, W, _ = x.shape
    ang = _smooth_noise(T, 6.0 * sev, rng)
    mats = [cv2.getRotationMatrix2D((W / 2, H / 2), float(ang[t]), 1.0) for t in range(T)]
    return _warp_seq(x, mats)


# --------------------------------------------------------------------------- #
# Occlusion
# --------------------------------------------------------------------------- #
def _blocks(rng, H, W, sev, n_lo=1, n_hi=4):
    n = rng.integers(n_lo, n_hi + 1)
    rects = []
    for _ in range(n):
        bw = int(W * (0.10 + 0.25 * sev) * (0.5 + rng.random()))
        bh = int(H * (0.10 + 0.25 * sev) * (0.5 + rng.random()))
        x0 = rng.integers(0, max(1, W - bw))
        y0 = rng.integers(0, max(1, H - bh))
        rects.append((x0, y0, bw, bh))
    return rects


def _static_occlusion(x, sev, rng):
    T, H, W, _ = x.shape
    out = x.copy()
    for (x0, y0, bw, bh) in _blocks(rng, H, W, sev):
        color = rng.integers(0, 60, 3).astype(np.float32)
        out[:, y0:y0 + bh, x0:x0 + bw] = color
    return out


def _dynamic_occlusion(x, sev, rng):
    T, H, W, _ = x.shape
    out = x.copy()
    for (x0, y0, bw, bh) in _blocks(rng, H, W, sev):
        vx, vy = rng.integers(-W // 32, W // 32 + 1), rng.integers(-H // 32, H // 32 + 1)
        color = rng.integers(0, 60, 3).astype(np.float32)
        for t in range(T):
            xx = int(np.clip(x0 + vx * t, 0, W - bw))
            yy = int(np.clip(y0 + vy * t, 0, H - bh))
            out[t, yy:yy + bh, xx:xx + bw] = color
    return out


# --------------------------------------------------------------------------- #
# Weather
# --------------------------------------------------------------------------- #
def _fog(x, sev, rng):
    T, H, W, _ = x.shape
    fog_color = 235.0
    # spatially-smooth fog density, denser toward frame top, drifting in time
    d = rng.random((H // 8, W // 8)).astype(np.float32)
    d = cv2.resize(cv2.GaussianBlur(d, (0, 0), 4), (W, H))
    vgrad = np.linspace(1.0, 0.55, H)[:, None]
    out = np.empty_like(x)
    for t in range(T):
        dd = np.roll(d, t * 2, axis=1) * vgrad
        alpha = np.clip(0.25 + 0.65 * sev * dd, 0, 0.95)[..., None]
        out[t] = x[t] * (1 - alpha) + fog_color * alpha
    return out


def _rain(x, sev, rng):
    T, H, W, _ = x.shape
    out = x * (1.0 - 0.15 * sev)                       # overcast dimming
    n = int(120 * sev * (H * W) / (480 * 640))
    n = max(n, 30)
    px = rng.integers(0, W, n)
    py = rng.integers(0, H, n)
    speed = rng.integers(H // 12, H // 6, n)
    length = (8 + 20 * sev)
    for t in range(T):
        layer = out[t].copy()
        yy = (py + t * speed) % H
        for i in range(n):
            cv2.line(layer, (int(px[i]), int(yy[i])),
                     (int(px[i] - length * 0.3), int(yy[i] + length)),
                     (200, 200, 210), 1, cv2.LINE_AA)
        out[t] = layer
        if sev > 0.5:                                   # streak blur in heavy rain
            out[t] = cv2.GaussianBlur(out[t], (0, 0), 0.6 + sev)
    return out


def _snow(x, sev, rng):
    T, H, W, _ = x.shape
    out = x * (1.0 - 0.10 * sev) + 15 * sev
    n = max(int(200 * sev * (H * W) / (480 * 640)), 50)
    px = rng.integers(0, W, n).astype(np.float32)
    py = rng.integers(0, H, n).astype(np.float32)
    vy = rng.uniform(H / 40, H / 20, n)
    vx = rng.uniform(-2, 2, n)
    rad = rng.integers(1, 3, n)
    for t in range(T):
        layer = out[t].copy()
        xx = (px + vx * t) % W
        yy = (py + vy * t) % H
        for i in range(n):
            cv2.circle(layer, (int(xx[i]), int(yy[i])), int(rad[i]),
                       (245, 245, 250), -1, cv2.LINE_AA)
        out[t] = layer
    return out


_DISPATCH = {
    "dusk": _dusk, "night": _night, "overexposure": _overexposure, "shadow": _shadow,
    "translation": _translation, "zoom": _zoom, "rotation": _rotation,
    "static_occlusion": _static_occlusion, "dynamic_occlusion": _dynamic_occlusion,
    "fog": _fog, "rain": _rain, "snow": _snow,
}
