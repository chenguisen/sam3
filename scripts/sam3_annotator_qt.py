#!/usr/bin/env python3
"""
SAM 3 珠光体分割标注工具 — Qt6 界面
====================================
功能: 画框标注 → SAM 3 全图自动分割 → 框选匹配 → 手动调整 → 保存结果

核心流程:
  1. 用户画框标记珠光体区域（1 个或多个）
  2. 点击 "▶ SAM 3 分割"
  3. SAM 3 全图一次推理，text prompt 指导分割
  4. 自动匹配每个用户框与最佳 mask
  5. 可选扩展全图相似实例

快捷键:
  Ctrl+滚轮: 缩放 | R: 重置缩放
  左键拖动: 画框 | 点击框/遮罩: 删除
  N/P: 上下翻页 | S: 保存 | Space/R: SAM 3 分割
"""

import sys, os, json, glob, gc
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QScrollArea, QFileDialog, QMessageBox,
    QStatusBar, QSpinBox, QSplitter, QListWidget, QFrame,
    QCheckBox,
)
from PyQt6.QtCore import Qt, QRectF, QPointF, QTimer
from PyQt6.QtGui import (QPolygonF,
    QPixmap, QImage, QPainter, QPen, QColor, QBrush, QAction,
    QKeySequence, QWheelEvent, QMouseEvent, QPaintEvent,
)

import torch
import torch.nn.functional as F

SAM3_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SAM3_ROOT)

# === Fix fused.py ===
import sam3.model.vitdet as _vd
def _fix_addmm(act, lin, m1):
    x = F.linear(m1, lin.weight, lin.bias)
    if act in [torch.nn.functional.relu, torch.nn.ReLU]: x = F.relu(x)
    elif act in [torch.nn.functional.gelu, torch.nn.GELU]: x = F.gelu(x)
    return x
_vd.addmm_act = _fix_addmm


def _xywh_to_normalized_cxcywh(box, image_width, image_height):
    """Convert an image-space XYWH rectangle to normalized CXCYWH."""
    x, y, w, h = box
    x1 = min(max(float(x), 0.0), float(image_width))
    y1 = min(max(float(y), 0.0), float(image_height))
    x2 = min(max(float(x + w), 0.0), float(image_width))
    y2 = min(max(float(y + h), 0.0), float(image_height))

    clipped_w = max(1.0, x2 - x1)
    clipped_h = max(1.0, y2 - y1)
    cx = x1 + clipped_w / 2.0
    cy = y1 + clipped_h / 2.0
    return [
        cx / float(image_width),
        cy / float(image_height),
        clipped_w / float(image_width),
        clipped_h / float(image_height),
    ]


def _box_iou_xyxy(box_a, box_b):
    """Compute IoU for two XYXY boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _xywh_to_xyxy(box):
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def _clip_box_xyxy(box_xyxy, image_shape):
    h, w = image_shape[:2]
    x1 = int(np.clip(np.floor(box_xyxy[0]), 0, max(0, w - 1)))
    y1 = int(np.clip(np.floor(box_xyxy[1]), 0, max(0, h - 1)))
    x2 = int(np.clip(np.ceil(box_xyxy[2]), x1 + 1, w))
    y2 = int(np.clip(np.ceil(box_xyxy[3]), y1 + 1, h))
    return x1, y1, x2, y2


def _extract_mask(mask_item) -> Optional[NDArray[np.bool_]]:
    if mask_item is None:
        return None
    if isinstance(mask_item, dict):
        mask_item = mask_item.get("mask")
    if mask_item is None:
        return None
    return np.asarray(mask_item, dtype=np.bool_)


def _mask_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}


def _mask_iou(mask_a, mask_b):
    inter = int(np.logical_and(mask_a, mask_b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(mask_a, mask_b).sum())
    if union == 0:
        return 0.0
    return inter / union


def _gray_image(image_array):
    if image_array.ndim == 2:
        return image_array.astype(np.float32)
    return image_array.astype(np.float32).mean(axis=2)


def _mask_mean_intensity(mask, gray_image):
    if int(mask.sum()) == 0:
        return 0.0
    return float(gray_image[mask].mean())


def _mask_descriptor(mask, image_array):
    gray_image = _gray_image(image_array)
    pixels = gray_image[mask]
    if pixels.size == 0:
        return {
            "hist": [0.0] * 8,
            "mean": 0.0,
            "std": 0.0,
            "area": 0.0,
            "aspect": 1.0,
        }

    hist, _ = np.histogram(pixels, bins=8, range=(0.0, 255.0), density=False)
    hist = hist.astype(np.float32)
    hist_sum = float(hist.sum())
    if hist_sum > 0:
        hist /= hist_sum

    bbox = _mask_bbox(mask)
    aspect = 1.0
    if bbox is not None and bbox["h"] > 0:
        aspect = float(bbox["w"]) / float(bbox["h"])

    return {
        "hist": hist.tolist(),
        "mean": float(pixels.mean()),
        "std": float(pixels.std()),
        "area": int(mask.sum()),
        "aspect": aspect,
    }


def _descriptor_similarity(desc_a, desc_b):
    hist_a = np.asarray(desc_a.get("hist", []), dtype=np.float32)
    hist_b = np.asarray(desc_b.get("hist", []), dtype=np.float32)
    if hist_a.size == 0 or hist_b.size == 0 or hist_a.shape != hist_b.shape:
        hist_score = 0.0
    else:
        hist_score = 1.0 - float(np.abs(hist_a - hist_b).sum()) / 2.0

    mean_a = float(desc_a.get("mean", 0.0))
    mean_b = float(desc_b.get("mean", 0.0))
    std_a = float(desc_a.get("std", 0.0))
    std_b = float(desc_b.get("std", 0.0))
    area_a = max(1.0, float(desc_a.get("area", 1.0)))
    area_b = max(1.0, float(desc_b.get("area", 1.0)))
    aspect_a = max(1e-6, float(desc_a.get("aspect", 1.0)))
    aspect_b = max(1e-6, float(desc_b.get("aspect", 1.0)))

    mean_score = max(0.0, 1.0 - abs(mean_a - mean_b) / max(20.0, abs(mean_a) * 0.35 + 1.0))
    std_score = max(0.0, 1.0 - abs(std_a - std_b) / max(20.0, abs(std_a) * 0.5 + 1.0))
    area_score = max(0.0, 1.0 - abs(np.log(area_a / area_b)) / np.log(2.5))
    aspect_score = max(0.0, 1.0 - abs(np.log(aspect_a / aspect_b)) / np.log(2.0))
    return 0.5 * hist_score + 0.2 * mean_score + 0.1 * std_score + 0.15 * area_score + 0.05 * aspect_score


def _resize_prob_map(mask_tensor, image_shape) -> NDArray[np.float32]:
    h, w = image_shape[:2]
    prob_map = mask_tensor.detach().cpu().numpy().squeeze()
    if prob_map.shape != (h, w):
        from skimage.transform import resize

        prob_map = resize(prob_map.astype(float), (h, w), preserve_range=True)
    return np.asarray(prob_map, dtype=np.float32)


def _otsu_threshold(values: NDArray[np.float32]) -> float:
    if values.size < 16:
        return 0.5
    values = values[np.isfinite(values)]
    if values.size < 16:
        return 0.5
    values = np.clip(values.astype(np.float32), 0.0, 1.0)
    if float(values.max()) - float(values.min()) < 1e-4:
        return float(values.mean())

    hist, bin_edges = np.histogram(values, bins=64, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * centers) / np.maximum(weight1, 1e-8)
    mean2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(weight2[::-1], 1e-8))[::-1]
    between = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    if between.size == 0:
        return 0.5
    return float(centers[int(np.argmax(between))])


def _adaptive_threshold_in_box(prob_map: NDArray[np.float32], user_box_xywh, image_shape) -> float:
    x1, y1, x2, y2 = _clip_box_xyxy(_xywh_to_xyxy(user_box_xywh), image_shape)
    roi = prob_map[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.5
    base = _otsu_threshold(roi.reshape(-1))
    q75 = float(np.quantile(roi, 0.75))
    threshold = max(base, q75 * 0.75)
    return float(np.clip(threshold, 0.2, 0.85))


def _encode_mask_rle(mask):
    flat = np.asarray(mask, dtype=np.uint8).ravel(order="C")
    counts = []
    last_value = 0
    run_length = 0
    for value in flat:
        value = int(value)
        if value == last_value:
            run_length += 1
        else:
            counts.append(run_length)
            run_length = 1
            last_value = value
    counts.append(run_length)
    return {
        "size": [int(mask.shape[0]), int(mask.shape[1])],
        "counts": counts,
        "order": "C",
    }


def _build_instance(mask, image_array, score=None, source_box=None, source_box_index=None, kind="seed"):
    mask_array = np.asarray(mask, dtype=np.bool_)
    area_pixels = int(mask_array.sum())
    total_area = int(mask_array.shape[0] * mask_array.shape[1])
    bbox = _mask_bbox(mask_array)
    descriptor = _mask_descriptor(mask_array, image_array)
    return {
        "mask": mask_array,
        "score": None if score is None else float(score),
        "kind": kind,
        "source_box_index": source_box_index,
        "source_box": None if source_box is None else {
            "x": float(source_box[0]),
            "y": float(source_box[1]),
            "w": float(source_box[2]),
            "h": float(source_box[3]),
        },
        "bbox": bbox,
        "area_pixels": area_pixels,
        "area_percentage": round(area_pixels / total_area * 100, 4) if total_area else 0.0,
        "mean_intensity": _mask_mean_intensity(mask_array, _gray_image(image_array)),
        "descriptor": descriptor,
    }


def _append_unique_instance(instances, candidate, iou_threshold=0.6):
    candidate_mask = _extract_mask(candidate)
    if candidate_mask is None or int(candidate_mask.sum()) == 0:
        return False
    for existing in instances:
        existing_mask = _extract_mask(existing)
        if existing_mask is None:
            continue
        if _mask_iou(existing_mask, candidate_mask) >= iou_threshold:
            return False
    instances.append(candidate)
    return True


def _mask_from_output(output, idx, image_shape, threshold=0.5):
    prob_tensors = output.get("masks_logits", output["masks"])
    prob_map = _resize_prob_map(prob_tensors[idx], image_shape)
    return np.asarray(prob_map >= threshold, dtype=np.bool_)



def _crop_box_region(box, image_shape, margin=0.25):
    h, w = image_shape[:2]
    x, y, bw, bh = box
    x1 = max(0, int(np.floor(x - bw * margin)))
    y1 = max(0, int(np.floor(y - bh * margin)))
    x2 = min(w, int(np.ceil(x + bw + bw * margin)))
    y2 = min(h, int(np.ceil(y + bh + bh * margin)))
    return x1, y1, x2, y2


def _map_crop_mask_to_full(mask, crop_bounds, image_shape):
    x1, y1, x2, y2 = crop_bounds
    full_mask = np.zeros(image_shape[:2], dtype=np.bool_)
    crop_h = max(0, y2 - y1)
    crop_w = max(0, x2 - x1)
    mapped_mask = mask
    if mapped_mask.shape != (crop_h, crop_w):
        from skimage.transform import resize

        mapped_mask = np.asarray(
            resize(mapped_mask.astype(float), (crop_h, crop_w), preserve_range=True) > 0.5,
            dtype=np.bool_,
        )
    full_mask[y1:y2, x1:x2] = mapped_mask[:crop_h, :crop_w]
    return full_mask


def _collect_similar_instances(scores_np, boxes_np, prob_maps, reference_instance, image_array, min_score, existing_instances):
    """Use one box-selected reference instance to retrieve similar masks on the full image."""
    reference_mask = _extract_mask(reference_instance)
    if reference_mask is None:
        return 0

    reference_area = max(1, int(reference_instance["area_pixels"]))
    reference_descriptor = reference_instance.get("descriptor", {})
    appended = 0

    for idx, score in enumerate(scores_np):
        score = float(score)
        if score < min_score:
            continue

        pred_box = boxes_np[idx]
        pred_area = max(0.0, pred_box[2] - pred_box[0]) * max(0.0, pred_box[3] - pred_box[1])
        if pred_area <= 0:
            continue
        box_area_ratio = pred_area / reference_area
        if box_area_ratio < 0.35 or box_area_ratio > 2.8:
            continue

        pred_box_xywh = [
            float(pred_box[0]),
            float(pred_box[1]),
            float(max(1.0, pred_box[2] - pred_box[0])),
            float(max(1.0, pred_box[3] - pred_box[1])),
        ]
        adaptive_threshold = _adaptive_threshold_in_box(
            prob_maps[idx],
            pred_box_xywh,
            image_array.shape,
        )
        candidate_mask = np.asarray(prob_maps[idx] >= adaptive_threshold, dtype=np.bool_)
        candidate_area = int(candidate_mask.sum())
        if candidate_area == 0:
            continue

        pixel_area_ratio = candidate_area / reference_area
        if pixel_area_ratio < 0.35 or pixel_area_ratio > 2.8:
            continue
        if _mask_iou(reference_mask, candidate_mask) >= 0.98:
            continue

        candidate_instance = _build_instance(
            candidate_mask,
            image_array=image_array,
            score=score,
            source_box=(
                reference_instance["source_box"]["x"],
                reference_instance["source_box"]["y"],
                reference_instance["source_box"]["w"],
                reference_instance["source_box"]["h"],
            ) if reference_instance.get("source_box") else None,
            source_box_index=reference_instance.get("source_box_index"),
            kind="similar",
        )
        similarity = _descriptor_similarity(
            reference_descriptor,
            candidate_instance.get("descriptor", {}),
        )
        if similarity < 0.68:
            continue
        candidate_instance["adaptive_score_threshold"] = round(float(adaptive_threshold), 4)
        candidate_instance["similarity"] = round(float(similarity), 4)
        if _append_unique_instance(existing_instances, candidate_instance, iou_threshold=0.65):
            appended += 1
    return appended




def _segment_instance_from_local_contrast(image_array, local_box_xywh):
    """Segment a bright local instance inside a user box without loading the heavy interactive model."""
    import cv2

    gray = _gray_image(image_array).astype(np.float32)
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = _clip_box_xyxy(_xywh_to_xyxy(local_box_xywh), image_array.shape)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None

    # Remove slow background variation and emphasize local bright structures.
    blur_size = max(5, int(min(h, w) * 0.08) | 1)
    local_bg = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    contrast = gray - local_bg
    contrast -= float(contrast.min())
    max_value = float(contrast.max())
    if max_value <= 1e-6:
        return None, None
    contrast = contrast / max_value

    roi_contrast = contrast[y1:y2, x1:x2]
    roi_uint8 = np.clip(roi_contrast * 255.0, 0, 255).astype(np.uint8)
    otsu_value, _ = cv2.threshold(roi_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_norm = float(otsu_value) / 255.0
    q95 = float(np.quantile(roi_contrast, 0.95))
    q99 = float(np.quantile(roi_contrast, 0.99))
    high_threshold = float(np.clip(max(otsu_norm * 0.8, q99 * 0.7, 0.18), 0.15, 0.82))
    low_threshold = float(np.clip(min(high_threshold * 0.65, max(q95 * 0.55, 0.08)), 0.06, high_threshold))

    high_mask = contrast >= high_threshold
    low_mask = contrast >= low_threshold
    full_box = np.zeros_like(high_mask, dtype=np.bool_)
    full_box[y1:y2, x1:x2] = True
    high_mask &= full_box
    low_mask &= full_box

    kernel = np.ones((3, 3), np.uint8)
    high_mask_u8 = cv2.morphologyEx(high_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    low_mask_u8 = cv2.morphologyEx(low_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    box_w = max(1.0, float(x2 - x1))
    box_h = max(1.0, float(y2 - y1))
    box_area = max(1.0, box_w * box_h)

    def _component_shape_score(component_mask):
        ys, xs = np.where(component_mask)
        if xs.size < 2:
            return 1.0, 0.0
        coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        cov = np.cov(coords.T)
        if cov.shape != (2, 2):
            return 1.0, 0.0
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-6)
        elongation = float(np.sqrt(eigvals[1] / eigvals[0]))
        major_vec = eigvecs[:, 1]
        angle = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])))
        return elongation, angle

    def _touches_box_border(component_mask):
        border_band = 2
        border = np.zeros_like(component_mask, dtype=np.bool_)
        border[y1:y1 + border_band, x1:x2] = True
        border[y2 - border_band:y2, x1:x2] = True
        border[y1:y2, x1:x1 + border_band] = True
        border[y1:y2, x2 - border_band:x2] = True
        return bool(np.any(component_mask & border))

    best_label = None
    best_key = None
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(high_mask_u8, connectivity=8)
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area <= 1:
            continue
        component_mask = labels == label_idx
        ys, xs = np.where(component_mask)
        if xs.size == 0:
            continue
        mean_contrast = float(contrast[component_mask].mean())
        area_ratio = area / box_area
        if area_ratio > 0.9:
            continue
        comp_cx = float(xs.mean())
        comp_cy = float(ys.mean())
        center_dist = np.hypot(comp_cx - center_x, comp_cy - center_y)
        center_score = 1.0 - center_dist / max(1.0, np.hypot(box_w, box_h) * 0.6)
        center_score = max(0.0, center_score)
        in_box_pixels = int(component_mask[y1:y2, x1:x2].sum())
        box_overlap = in_box_pixels / max(1.0, area)
        elongation, _ = _component_shape_score(component_mask)
        border_penalty = 0.0 if not _touches_box_border(component_mask) else 1.0
        candidate_key = (
            box_overlap,
            center_score,
            min(elongation, 6.0),
            mean_contrast,
            -border_penalty,
            -abs(area_ratio - 0.08),
        )
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_label = label_idx

    if best_label is not None:
        seed_mask = np.asarray(labels == best_label, dtype=np.uint8)
    else:
        # Fallback: use the brightest pixel inside the box as the seed.
        peak_y, peak_x = np.unravel_index(np.argmax(roi_contrast), roi_contrast.shape)
        seed_mask = np.zeros_like(low_mask_u8, dtype=np.uint8)
        seed_mask[y1 + peak_y, x1 + peak_x] = 1

    grown_seed = cv2.dilate(seed_mask, kernel, iterations=1)
    low_labels_count, low_labels, _, _ = cv2.connectedComponentsWithStats(low_mask_u8, connectivity=8)
    best_low_label = None
    best_angle = 0.0
    best_low_key = None
    for label_idx in range(1, low_labels_count):
        component_mask = low_labels == label_idx
        overlap = int((component_mask & (grown_seed > 0)).sum())
        if overlap <= 0:
            continue
        ys, xs = np.where(component_mask)
        if xs.size == 0:
            continue
        area = int(component_mask.sum())
        comp_cx = float(xs.mean())
        comp_cy = float(ys.mean())
        center_dist = np.hypot(comp_cx - center_x, comp_cy - center_y)
        center_score = 1.0 - center_dist / max(1.0, np.hypot(box_w, box_h) * 0.6)
        center_score = max(0.0, center_score)
        elongation, angle = _component_shape_score(component_mask)
        border_penalty = 0.0 if not _touches_box_border(component_mask) else 1.0
        area_ratio = area / box_area
        key = (
            overlap,
            center_score,
            min(elongation, 8.0),
            -border_penalty,
            -abs(area_ratio - 0.08),
        )
        if best_low_key is None or key > best_low_key:
            best_low_key = key
            best_low_label = label_idx
            best_angle = angle

    if best_low_label is None:
        return None, None

    best_mask = np.asarray(low_labels == best_low_label, dtype=np.bool_)
    line_kernel_size = max(5, int(round(min(box_w, box_h) * 0.35)) | 1)
    line_kernel = np.zeros((line_kernel_size, line_kernel_size), dtype=np.uint8)
    center = line_kernel_size // 2
    cv2.line(line_kernel, (0, center), (line_kernel_size - 1, center), 1, 1)
    rotate_mat = cv2.getRotationMatrix2D((center, center), best_angle, 1.0)
    oriented_kernel = cv2.warpAffine(
        line_kernel,
        rotate_mat,
        (line_kernel_size, line_kernel_size),
        flags=cv2.INTER_NEAREST,
    )
    # Resize kernel to match oriented_kernel shape before element-wise operation
    if kernel.shape != oriented_kernel.shape:
        resized_kernel = cv2.resize(kernel.astype(np.uint8), (oriented_kernel.shape[1], oriented_kernel.shape[0]),
                                    interpolation=cv2.INTER_NEAREST) > 0
        oriented_kernel = np.maximum(oriented_kernel, resized_kernel.astype(np.uint8))
    else:
        oriented_kernel = np.maximum(oriented_kernel, kernel)
    best_mask = cv2.morphologyEx(best_mask.astype(np.uint8), cv2.MORPH_CLOSE, oriented_kernel) > 0
    best_mask &= full_box
    score = float(best_key[1]) if best_key is not None else None
    return np.asarray(best_mask, dtype=np.bool_), score


def _clear_box_prompt_state(state):
    """Drop per-prompt tensors while keeping the encoded image and text features."""
    for key in ("geometric_prompt", "boxes", "masks", "masks_logits", "scores"):
        state.pop(key, None)


class ImageCanvas(QWidget):
    """图像显示组件，支持缩放和标注"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self.dragging = False
        self.drag_start = None
        self.draw_start = None
        self.draw_rect = None  # (x, y, w, h) in image coords
        self.boxes = []        # list of (x, y, w, h) in image coords
        self.masks = []        # list of dicts with "mask" key, or raw 2D bool arrays
        self.mask_infos = []   # list of metadata for masks
        self.current_image = None  # numpy array
        self.setMouseTracking(True)
        self.setMinimumSize(600, 400)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def load_image(self, img_array):
        """加载 numpy 图像，自动适配窗口大小"""
        self.current_image = img_array
        h, w = img_array.shape[:2]
        if img_array.dtype != np.uint8:
            img_array = (img_array.clip(0, 255)).astype(np.uint8)
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=2)
        qimg = QImage(img_array.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimg)
        # 自动适配窗口大小并居中
        self.scale = 1.0
        cw = self.width() - 20
        ch = self.height() - 20
        if cw > 0 and ch > 0 and w > 0 and h > 0:
            self.scale = min(cw / w, ch / h, 1.0)
            # 居中显示
            self.offset = QPointF((cw - w * self.scale) / 2, (ch - h * self.scale) / 2)
        else:
            self.offset = QPointF(0, 0)
        self.boxes = []
        self.masks = []
        self.mask_infos = []
        self.update()

    def _img_to_view(self, px, py):
        """图像坐标 → 控件坐标"""
        return px * self.scale + self.offset.x(), py * self.scale + self.offset.y()

    def _view_to_img(self, vx, vy):
        """控件坐标 → 图像坐标"""
        return (vx - self.offset.x()) / self.scale, (vy - self.offset.y()) / self.scale

    def paintEvent(self, event):
        """绘制"""
        if self.pixmap is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # 背景
        painter.fillRect(self.rect(), QColor(40, 40, 40))

        # 图像
        painter.translate(self.offset)
        painter.scale(self.scale, self.scale)
        painter.drawPixmap(0, 0, self.pixmap)

        # 画遮罩（半透明填充 + 轮廓）
        for i, mask in enumerate(self.masks):
            mask_array = _extract_mask(mask)
            if mask_array is None or int(mask_array.sum()) == 0:
                continue
            if mask_array.ndim != 2:
                continue

            fill_color = QColor.fromHsv((i * 37) % 360, 180, 255, 60)
            overlay = np.zeros((mask_array.shape[0], mask_array.shape[1], 4), dtype=np.uint8)
            overlay[mask_array] = [
                fill_color.red(),
                fill_color.green(),
                fill_color.blue(),
                fill_color.alpha(),
            ]
            overlay_img = QImage(
                overlay.data,
                overlay.shape[1],
                overlay.shape[0],
                overlay.shape[1] * 4,
                QImage.Format.Format_RGBA8888,
            ).copy()
            painter.drawImage(0, 0, overlay_img)

            # 轮廓
            color2 = QColor.fromHsv((i * 37) % 360, 180, 255, 200)
            painter.setPen(QPen(color2, max(2.0, 2 / self.scale)))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            try:
                from skimage.measure import find_contours
                for contour in find_contours(mask_array.astype(float), 0.5):
                    if len(contour) < 3: continue
                    step = max(1, len(contour) // 100)
                    pts = [QPointF(float(p[1]), float(p[0])) for p in contour[::step]]
                    painter.drawPolygon(QPolygonF(pts))
            except ImportError:
                pass

        # 画框
        painter.setPen(QPen(QColor(0, 255, 0), 2 / self.scale))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for x, y, w, h in self.boxes:
            painter.drawRect(QRectF(x, y, w, h))

        # 画正在拖动的框
        if self.draw_rect is not None:
            x, y, w, h = self.draw_rect
            painter.setPen(QPen(QColor(255, 255, 0), 2 / self.scale, Qt.PenStyle.DashLine))
            painter.drawRect(QRectF(x, y, w, h))

        painter.end()

    def wheelEvent(self, event: QWheelEvent):
        """滚轮缩放"""
        old_scale = self.scale
        if event.angleDelta().y() > 0:
            self.scale *= 1.15
        else:
            self.scale /= 1.15
        self.scale = max(0.1, min(10.0, self.scale))

        # 以鼠标位置为中心缩放
        mx, my = event.position().x(), event.position().y()
        img_x, img_y = self._view_to_img(mx, my)
        self.offset.setX(mx - img_x * self.scale)
        self.offset.setY(my - img_y * self.scale)
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if self.pixmap is None:
            return
        vx, vy = event.position().x(), event.position().y()
        ix, iy = self._view_to_img(vx, vy)

        if event.button() == Qt.MouseButton.LeftButton:
            # 检查是否点击了遮罩
            for i in range(len(self.masks) - 1, -1, -1):
                mask = _extract_mask(self.masks[i])
                if mask is not None and 0 <= int(iy) < mask.shape[0] and 0 <= int(ix) < mask.shape[1]:
                    if mask[int(iy), int(ix)]:
                        self.masks.pop(i)
                        if i < len(self.mask_infos):
                            self.mask_infos.pop(i)
                        self.update()
                        if self.parent() and hasattr(self.parent(), 'on_masks_changed'):
                            self.parent().on_masks_changed(i)
                        return

            # 检查是否点击了已有框
            for i, (x, y, w, h) in enumerate(self.boxes):
                if x <= ix <= x + w and y <= iy <= y + h:
                    self.boxes.pop(i)
                    self.update()
                    if self.parent() and hasattr(self.parent(), 'on_boxes_changed'):
                        self.parent().on_boxes_changed()
                    return

            # 开始画框
            self.draw_start = (ix, iy)
        elif event.button() == Qt.MouseButton.MiddleButton:
            # 中键拖动平移
            self.dragging = True
            self.drag_start = event.position()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.draw_start is not None:
            vx, vy = event.position().x(), event.position().y()
            ix, iy = self._view_to_img(vx, vy)
            sx, sy = self.draw_start
            x, y = min(sx, ix), min(sy, iy)
            w, h = abs(ix - sx), abs(iy - sy)
            self.draw_rect = (x, y, w, h)
            self.update()
        elif self.dragging:
            delta = event.position() - self.drag_start
            self.offset += delta
            self.drag_start = event.position()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.draw_start is not None:
            vx, vy = event.position().x(), event.position().y()
            ix, iy = self._view_to_img(vx, vy)
            sx, sy = self.draw_start
            w, h = abs(ix - sx), abs(iy - sy)
            if w > 5 and h > 5:
                x, y = min(sx, ix), min(sy, iy)
                self.boxes.append((x, y, w, h))
                # 发出信号通知父窗口
                if self.parent() and hasattr(self.parent(), 'on_boxes_changed'):
                    self.parent().on_boxes_changed()
            self.draw_start = None
            self.draw_rect = None
            self.update()
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.dragging = False

    def reset_view(self):
        """重置缩放"""
        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM 3 SAED 标注工具")
        self.resize(1400, 900)

        # 数据
        self.data_dir = os.path.join(SAM3_ROOT, "data", "3")
        self.output_dir = os.path.join(self.data_dir, "annotations")
        os.makedirs(self.output_dir, exist_ok=True)
        self.image_paths = []
        self.current_idx = 0
        self.model = None
        self.processor = None
        self.device = "cpu"

        # UI
        self._setup_ui()
        self._load_images()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # 左侧：画布
        self.canvas = ImageCanvas()
        layout.addWidget(self.canvas, 1)

        # 右侧面板
        panel = QFrame()
        panel.setFixedWidth(250)
        panel_layout = QVBoxLayout(panel)

        # 图片列表
        panel_layout.addWidget(QLabel("图片列表:"))
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self._on_select)
        panel_layout.addWidget(self.list_widget)

        # 信息
        self.info_label = QLabel("框: 0 | 分割: 0")
        panel_layout.addWidget(self.info_label)

        # 按钮
        for text, slot in [
            ("📂 打开目录", self._open_dir),
            ("▶ SAM 3 分割", self._run_sam3),
            ("🗑 清除框", self._clear_boxes),
            ("🗑 清除分割", self._clear_masks),
            ("🔍 重置视图", lambda: self.canvas.reset_view()),
            ("💾 保存", self._save),
            ("⏭ 下一张", lambda: self._navigate(1)),
            ("⏮ 上一张", lambda: self._navigate(-1)),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            panel_layout.addWidget(btn)

        # 阈值
        panel_layout.addWidget(QLabel("置信度阈值:"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1, 99)
        self.threshold_spin.setValue(5)  # 0.10
        panel_layout.addWidget(self.threshold_spin)

        self.expand_similar_checkbox = QCheckBox("单框扩展全图相似实例")
        self.expand_similar_checkbox.setChecked(False)
        panel_layout.addWidget(self.expand_similar_checkbox)

        panel_layout.addStretch()
        layout.addWidget(panel)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

        # 快捷键
        QAction("Next", self, shortcut=QKeySequence("n"), triggered=lambda: self._navigate(1))
        QAction("Prev", self, shortcut=QKeySequence("p"), triggered=lambda: self._navigate(-1))
        QAction("Save", self, shortcut=QKeySequence("s"), triggered=self._save)
        QAction("Run SAM3", self, shortcut=QKeySequence("r"), triggered=self._run_sam3)
        QAction("Reset Zoom", self, shortcut=QKeySequence("R"), triggered=lambda: self.canvas.reset_view())

    def _load_images(self):
        """扫描图片"""
        exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
        self.image_paths = sorted(set(
            p for ext in exts
            for p in glob.glob(os.path.join(self.data_dir, f"*{ext}"))
            + glob.glob(os.path.join(self.data_dir, f"*{ext.upper()}"))
        ))
        self.list_widget.clear()
        for p in self.image_paths:
            self.list_widget.addItem(os.path.basename(p))
        if self.image_paths:
            self.list_widget.setCurrentRow(0)

    def _on_select(self, idx):
        if 0 <= idx < len(self.image_paths):
            self.current_idx = idx
            img_pil = Image.open(self.image_paths[idx]).convert("RGB")
            # 缩放到 SAM 3 内部尺寸（省显存）
            if max(img_pil.size) > 1008:
                img_pil.thumbnail((1008, 1008), Image.LANCZOS)
            img = np.array(img_pil)
            self.canvas.load_image(img)
            self._update_info()
            self.status_bar.showMessage(os.path.basename(self.image_paths[idx]))

    def _navigate(self, delta):
        idx = self.current_idx + delta
        if 0 <= idx < len(self.image_paths):
            self.list_widget.setCurrentRow(idx)

    def _open_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择图片目录", self.data_dir)
        if d:
            self.data_dir = d
            self.output_dir = os.path.join(d, "annotations")
            os.makedirs(self.output_dir, exist_ok=True)
            self._load_images()

    def _update_info(self):
        n_boxes = len(self.canvas.boxes)
        n_masks = len(self.canvas.masks)
        self.info_label.setText(f"框: {n_boxes} | 分割: {n_masks}")

    def on_boxes_changed(self):
        self._update_info()

    def on_masks_changed(self, deleted_idx=None):
        self._update_info()
        if deleted_idx is not None:
            self.status_bar.showMessage(f"已删除分割实例 #{deleted_idx + 1}")

    def _clear_boxes(self):
        self.canvas.boxes.clear()
        self.canvas.update()
        self._update_info()

    def _clear_masks(self):
        self.canvas.masks.clear()
        self.canvas.mask_infos.clear()
        self.canvas.update()
        self._update_info()

    def _get_model(self):
        if self.model is None:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            ckpt = os.path.join(SAM3_ROOT, "checkpoints", "sam3.pt")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.status_bar.showMessage("构建 SAM 3 模型...")
            QApplication.processEvents()
            self.model = build_sam3_image_model(
                checkpoint_path=ckpt, device=self.device,
                eval_mode=True, enable_segmentation=True, enable_inst_interactivity=False,
            )
            self.processor = Sam3Processor(self.model, resolution=1008, device=self.device, confidence_threshold=0.0)
            self.status_bar.showMessage("模型就绪")
        return self.model, self.processor, self.device

    def _run_sam3(self):
        if self.canvas.current_image is None:
            self.status_bar.showMessage("请先加载图片")
            return

        if not self.canvas.boxes:
            self.status_bar.showMessage("请先画框")
            return

        self.status_bar.showMessage("SAM 3 分割中...")
        QApplication.processEvents()

        model, processor, device = self._get_model()
        img_array = self.canvas.current_image
        img_pil = Image.fromarray(img_array)

        threshold = self.threshold_spin.value() / 100.0
        boxes_to_process = list(self.canvas.boxes)
        all_instances = []
        state = None

        try:
            if device == "cuda":
                torch.cuda.empty_cache()

            # --- 对每个框: 裁剪 ROI → SAM3 box-prompt 分割 ---
            for idx, user_box in enumerate(boxes_to_process):
                self.status_bar.showMessage(
                    f"处理框 {idx+1}/{len(boxes_to_process)}..."
                )
                QApplication.processEvents()

                # 裁剪 ROI（精确对应用户框）
                crop_bounds = _crop_box_region(user_box, img_array.shape, margin=0.0)
                x1, y1, x2, y2 = crop_bounds
                crop_img = img_pil.crop((x1, y1, x2, y2))
                crop_arr = np.array(crop_img)
                crop_h, crop_w = crop_arr.shape[:2]

                # 框在裁剪图中的局部坐标
                local_box_xywh = (
                    user_box[0] - x1,
                    user_box[1] - y1,
                    user_box[2],
                    user_box[3],
                )

                # 缩放到 SAM3 尺寸（省显存）
                scale_x = 1.0
                scale_y = 1.0
                if max(crop_img.size) > 1008:
                    ow, oh = crop_img.size
                    crop_img.thumbnail((1008, 1008), Image.LANCZOS)
                    scale_x = crop_img.size[0] / ow
                    scale_y = crop_img.size[1] / oh

                # 转换 box 到 normalized CxCyWH
                cw, ch = crop_img.size
                scaled_local_box = (
                    local_box_xywh[0] * scale_x,
                    local_box_xywh[1] * scale_y,
                    local_box_xywh[2] * scale_x,
                    local_box_xywh[3] * scale_y,
                )
                box_norm = _xywh_to_normalized_cxcywh(
                    scaled_local_box, cw, ch
                )

                # SAM3 box-prompt 推理
                state = processor.set_image(crop_img)
                output_state = processor.add_geometric_prompt(
                    box=box_norm, label=True, state=state,
                )

                if "masks" not in output_state or output_state["masks"].numel() == 0:
                    # fallback 到局部对比度
                    local_mask, local_score = _segment_instance_from_local_contrast(
                        crop_arr, local_box_xywh,
                    )
                    if local_mask is not None:
                        full_mask = _map_crop_mask_to_full(
                            local_mask, crop_bounds, img_array.shape,
                        )
                        instance = _build_instance(
                            full_mask, image_array=img_array,
                            score=local_score or 0.5,
                            source_box=user_box, source_box_index=idx,
                            kind="seed",
                        )
                        instance["adaptive_score_threshold"] = round(float(threshold), 4)
                        _append_unique_instance(all_instances, instance, iou_threshold=0.6)
                    continue

                # 取 mask（多候选取最高分）
                masks_tensor = output_state.get("masks_logits", output_state["masks"])
                scores_np = output_state["scores"].detach().cpu().numpy()
                best_idx = int(np.argmax(scores_np))
                best_score = float(scores_np[best_idx])

                # 解码 mask
                prob_map = _resize_prob_map(masks_tensor[best_idx], crop_arr.shape)
                adaptive_th = _adaptive_threshold_in_box(
                    prob_map, local_box_xywh, crop_arr.shape,
                )
                mask = np.asarray(prob_map >= adaptive_th, dtype=np.bool_)

                # 清理 GPU state（每框释放）
                _clear_box_prompt_state(state)
                del state, output_state, masks_tensor
                state = None
                if device == "cuda":
                    torch.cuda.empty_cache()

                if int(mask.sum()) > 0:
                    full_mask = _map_crop_mask_to_full(
                        mask, crop_bounds, img_array.shape,
                    )
                    instance = _build_instance(
                        full_mask, image_array=img_array,
                        score=best_score,
                        source_box=user_box, source_box_index=idx,
                        kind="seed",
                    )
                    instance["adaptive_score_threshold"] = round(float(threshold), 4)
                    _append_unique_instance(all_instances, instance, iou_threshold=0.6)
                    print(f"[DEBUG] 框 {idx+1}: SAM3 box-prompt 成功, score={best_score:.4f}")
                else:
                    # fallback
                    local_mask, local_score = _segment_instance_from_local_contrast(
                        crop_arr, local_box_xywh,
                    )
                    if local_mask is not None:
                        full_mask = _map_crop_mask_to_full(
                            local_mask, crop_bounds, img_array.shape,
                        )
                        instance = _build_instance(
                            full_mask, image_array=img_array,
                            score=local_score or 0.5,
                            source_box=user_box, source_box_index=idx,
                            kind="seed",
                        )
                        instance["adaptive_score_threshold"] = round(float(threshold), 4)
                        _append_unique_instance(all_instances, instance, iou_threshold=0.6)

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                self.status_bar.showMessage("⚠️ 显存不足，已清理缓存，请重试")
                return
            raise
        finally:
            if state is not None:
                _clear_box_prompt_state(state)
            del state
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        if all_instances:
            self.canvas.masks = all_instances
            self.canvas.mask_infos = []
            self.canvas.update()
            self._update_info()
            self.status_bar.showMessage(
                f"✅ {len(all_instances)}/{len(boxes_to_process)} 个框分割成功"
            )
        else:
            self.status_bar.showMessage("⚠️ 无结果，画大一点的框或降低阈值试试")

    def _save(self):
        if self.canvas.current_image is None:
            return
        name = os.path.basename(self.image_paths[self.current_idx])
        h, w = self.canvas.current_image.shape[:2]
        total_area = h * w

        union_mask = np.zeros((h, w), dtype=np.bool_)
        instances_payload = []
        for idx, instance in enumerate(self.canvas.masks, start=1):
            mask = _extract_mask(instance)
            if mask is None:
                continue
            union_mask |= mask
            payload = {
                "id": idx,
                "kind": instance.get("kind", "mask") if isinstance(instance, dict) else "mask",
                "score": None if not isinstance(instance, dict) else instance.get("score"),
                "threshold": None if not isinstance(instance, dict) else instance.get("adaptive_score_threshold"),
                "area_pixels": int(mask.sum()),
                "area_percentage": round(float(mask.sum()) / total_area * 100, 4) if total_area > 0 else 0.0,
                "bbox": _mask_bbox(mask),
                "source_box_index": None if not isinstance(instance, dict) else instance.get("source_box_index"),
                "source_box": None if not isinstance(instance, dict) else instance.get("source_box"),
                "mean_intensity": None if not isinstance(instance, dict) else instance.get("mean_intensity"),
                "selection_metrics": None if not isinstance(instance, dict) else instance.get("selection_metrics"),
                "mask_rle": _encode_mask_rle(mask),
            }
            instances_payload.append(payload)

        mask_area = int(union_mask.sum())
        area_pct = float(mask_area) / total_area * 100 if total_area > 0 else 0

        data = {
            "image": name,
            "image_size": (w, h),
            "n_boxes": len(self.canvas.boxes),
            "n_masks": len(instances_payload),
            "area_total_pixels": total_area,
            "area_mask_pixels": int(mask_area),
            "area_percentage": round(area_pct, 4),
            "boxes": [{"x": b[0], "y": b[1], "w": b[2], "h": b[3]} for b in self.canvas.boxes],
            "instances": instances_payload,
        }

        save_path = os.path.join(self.output_dir, f"{os.path.splitext(name)[0]}.json")
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
        self.status_bar.showMessage(f"✅ 已保存: {save_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
