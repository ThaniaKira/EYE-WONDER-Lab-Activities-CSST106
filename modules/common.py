"""Shared processing / visualization helpers for the VisionLab modules."""
from __future__ import annotations
import base64
import io
from typing import Any
import cv2
import numpy as np


def decode_image(file_storage, *, max_side=1800):
    if not file_storage or not getattr(file_storage, 'filename', ''):
        raise ValueError('Please upload an image first or use the sample flower image.')
    raw = file_storage.read()
    if not raw:
        raise ValueError('The uploaded image is empty.')
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('The file is not a supported image. Try JPG, PNG, BMP, or WEBP.')
    h, w = image.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        image = cv2.resize(image, (max(1, round(w*s)), max(1, round(h*s))), interpolation=cv2.INTER_AREA)
    return image


def to_data_url(image, *, fmt='.png') -> str:
    if image.ndim == 2 and image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if image.ndim == 3 and image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    ok, buffer = cv2.imencode(fmt, image)
    if not ok:
        raise ValueError('Could not generate an image preview.')
    media_type = 'image/png' if fmt == '.png' else 'image/jpeg'
    return f'data:{media_type};base64,' + base64.b64encode(buffer).decode('ascii')


def output(label, image, note='') -> dict[str, Any]:
    return {'label': label, 'src': to_data_url(image), 'note': note}


def num(values, name, default, minimum=None, maximum=None, cast=float):
    try:
        v = cast(values.get(name, default))
    except (ValueError, TypeError):
        v = cast(default)
    if minimum is not None:
        v = max(minimum, v)
    if maximum is not None:
        v = min(maximum, v)
    return v


def hist_image(images: list[tuple[str, np.ndarray]]):
    """Lightweight, readable histogram drawn by OpenCV (no pyplot backend dependency)."""
    w, h = 820, 300
    canvas = np.full((h, w, 3), (251, 246, 255), dtype=np.uint8)
    margin_l, margin_r, margin_t, margin_b = 55, 25, 24, 42
    plot_w, plot_h = w - margin_l - margin_r, h - margin_t - margin_b
    for x in range(0, 256, 51):
        xx = margin_l + round(x / 255 * plot_w)
        cv2.line(canvas, (xx, margin_t), (xx, h - margin_b), (234, 222, 244), 1)
        cv2.putText(canvas, str(x), (xx-10, h-14), cv2.FONT_HERSHEY_SIMPLEX, .4, (94, 64, 120), 1, cv2.LINE_AA)
    for y in range(0, 5):
        yy = margin_t + round(y * plot_h / 4)
        cv2.line(canvas, (margin_l, yy), (w-margin_r, yy), (234, 222, 244), 1)
    palette = [(204, 99, 152), (177, 84, 113), (94, 116, 226), (151, 92, 227)]
    for i, (name, gray) in enumerate(images):
        vals = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        vals = np.sqrt(vals)
        vmax = np.max(vals) or 1
        pts = np.column_stack((margin_l + np.arange(256)*plot_w/255, margin_t + plot_h*(1-vals/vmax))).astype(np.int32)
        cv2.polylines(canvas, [pts], False, palette[i%len(palette)], 2, cv2.LINE_AA)
        cv2.circle(canvas, (margin_l + i*180, 12), 5, palette[i%len(palette)], -1, cv2.LINE_AA)
        cv2.putText(canvas, name[:19], (margin_l+12+i*180, 17), cv2.FONT_HERSHEY_SIMPLEX, .43, (68, 43, 82), 1, cv2.LINE_AA)
    return canvas


def base_result(img, lab_title, *, note=''):
    h, w = img.shape[:2]
    return {
        'title': lab_title,
        'outputs': [output('Original image', img)],
        'stats': [{'label': 'Width', 'value': f'{w}px'}, {'label': 'Height', 'value': f'{h}px'}, {'label': 'Color channels', 'value': '3 (BGR)'}],
        'note': note,
    }
