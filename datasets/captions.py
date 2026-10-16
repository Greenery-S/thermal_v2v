"""Structured captions: fixed-order English sentences, one sentence per field, in the descriptive style Wan2.2 was
trained on. Agreed format (2026-10-16): task + sensor + media (video vs. still image repeated into a clip).
No scene, no event, no fps, no per-field dropout; only whole-caption dropout ("" = CFG unconditional) in the trainer.
Later fields ("The scene shows ..." / "Thermal event: ...") can be appended without changing these sentences.

  caption("rgb2thermal", "radiometric", "video")
  -> "A thermal infrared video converted from an RGB video. Captured by a radiometric thermal camera.
      Consecutive frames from a real video."
  caption("rgb2thermal", "agc8", "image")
  -> "A thermal infrared video converted from an RGB video. Captured by an 8-bit auto-gain thermal camera.
      A single still image repeated over time."
"""
from __future__ import annotations

from itertools import product
from typing import List

TASK = {
    "rgb2thermal": "A thermal infrared video converted from an RGB video.",
    "thermal2rgb": "An RGB video converted from a thermal infrared video.",
    "joint": "A paired RGB and thermal infrared video.",
}
SENSOR = {
    "radiometric": "Captured by a radiometric thermal camera.",   # 16-bit, fixed per-clip normalisation (TartanRGBT)
    "agc8": "Captured by an 8-bit auto-gain thermal camera.",     # 8-bit per-frame auto gain (KAIST, LLVIP)
}
MEDIA = {
    "video": "Consecutive frames from a real video.",
    "image": "A single still image repeated over time.",          # clips made by repeating one still image
}


def caption(direction: str, sensor: str, media: str = "video") -> str:
    return " ".join([TASK[direction], SENSOR[sensor], MEDIA[media]])


def all_captions() -> List[str]:
    """Every string the trainer can ask for, plus "" (unconditional). Pre-encoded with UMT5 once."""
    return sorted({""} | {caption(d, s, m) for d, s, m in product(TASK, SENSOR, MEDIA)})


if __name__ == "__main__":
    for c in all_captions():
        print(repr(c))
