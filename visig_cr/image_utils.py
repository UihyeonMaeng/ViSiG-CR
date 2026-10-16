"""Canonical image loading and fixed-range bit-depth normalization.

The ReXGradient PNG release contains 16-bit grayscale images.  Pillow's
direct ``I;16 -> RGB`` conversion saturates values above 255, which turns a
typical radiograph into an almost entirely white model input.  Every ViSiG-CR
image consumer must therefore pass through the helpers in this module before
converting an image to RGB.

The policy is intentionally deterministic: unsigned 16-bit samples are
mapped from the full stored range 0..65535 to 0..255 by nearest-integer
division by 257.  It performs neither per-image min/max stretching nor
photometric inversion, so the same source pixels receive the same values in
the reader, retrieval encoders, comparison models, and materialized proxies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_NORMALIZATION_POLICY = "fixed_uint16_to_uint8_no_window_v1"
UINT16_MAX = 65_535
UINT8_MAX = 255
UINT16_TO_UINT8_DIVISOR = 257


def image_normalization_provenance() -> dict[str, Any]:
    """Return a fresh JSON-serializable description of the active policy."""

    return {
        "policy_id": IMAGE_NORMALIZATION_POLICY,
        "uint16_input_range": [0, UINT16_MAX],
        "uint8_output_range": [0, UINT8_MAX],
        "uint16_mapping": "round(value / 257)",
        "per_image_min_max_stretching": False,
        "photometric_inversion": False,
    }


def normalize_image_to_uint8(source: Image.Image) -> Image.Image:
    """Return a detached image whose samples are safe for RGB conversion.

    Existing 8-bit images are copied without changing their mode or palette.
    This preserves the established behavior for ordinary PNG/JPEG inputs.  Integer
    images with samples above 255 and within the 16-bit range use the same
    fixed-range conversion as native uint16 images.  Unsupported ranges and
    non-finite floating-point samples fail explicitly rather than silently
    producing an unusable model input.
    """

    if not isinstance(source, Image.Image):
        raise TypeError("source must be a PIL.Image.Image")

    source.load()
    array = np.asarray(source)
    if array.size == 0:
        raise ValueError("Image contains no pixels")

    # Preserve modes such as RGB, RGBA, L, and P exactly.  In particular, a
    # palette image must not be reconstructed from its uint8 palette indices.
    if array.dtype == np.uint8:
        return source.copy()

    if array.dtype == np.bool_:
        return Image.fromarray(array.astype(np.uint8) * UINT8_MAX)

    if np.issubdtype(array.dtype, np.integer):
        minimum = int(array.min())
        maximum = int(array.max())
        native_uint16 = (
            np.issubdtype(array.dtype, np.unsignedinteger)
            and array.dtype.itemsize == 2
        )

        if native_uint16 or maximum > UINT8_MAX:
            if minimum < 0 or maximum > UINT16_MAX:
                raise ValueError(
                    "Integer image exceeds the supported unsigned 16-bit range: "
                    f"min={minimum}, max={maximum}, dtype={array.dtype}"
                )
            converted = (
                (array.astype(np.uint32) + 128) // UINT16_TO_UINT8_DIVISOR
            ).astype(np.uint8)
        else:
            converted = np.clip(array, 0, UINT8_MAX).astype(np.uint8)

        return Image.fromarray(converted)

    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("Floating-point image contains NaN or infinity")
        return Image.fromarray(
            np.clip(array, 0.0, float(UINT8_MAX)).astype(np.uint8)
        )

    raise TypeError(f"Unsupported image dtype: {array.dtype}")


def load_uint8_image(path: str | Path) -> Image.Image:
    """Load *path* into a detached image using the canonical policy."""

    with Image.open(path) as source:
        return normalize_image_to_uint8(source)


def load_rgb_image(path: str | Path) -> Image.Image:
    """Load *path* as a detached RGB image using the canonical policy."""

    normalized = load_uint8_image(path)
    try:
        return normalized.convert("RGB")
    finally:
        normalized.close()


__all__ = [
    "IMAGE_NORMALIZATION_POLICY",
    "UINT16_MAX",
    "UINT16_TO_UINT8_DIVISOR",
    "UINT8_MAX",
    "image_normalization_provenance",
    "load_rgb_image",
    "load_uint8_image",
    "normalize_image_to_uint8",
]
