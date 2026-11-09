"""Image and filesystem helpers.

OpenCV on Windows cannot open paths containing non-ASCII characters
``cv2.imread`` silently returns ``None`` for e.g. ``D:\\微流控原始数据\\...``.
Every image therefore goes through :func:`imread_unicode`, which falls back to
``np.fromfile`` + ``cv2.imdecode``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: str | Path) -> np.ndarray | None:
    """Read a BGR image, tolerating non-ASCII paths (Windows/OpenCV quirk)."""
    path = Path(path)
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def ensure_dirs(*paths: str | Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
