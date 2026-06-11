
 """
SAM 3 SAED Diffraction Spot Detection
======================================
自包含函数：输入 SAED 衍射图像路径，返回检测到的衍射斑点信息。

使用方式:
    from sam3.sam3_detect_saed_spots import detect_saed_spots

    spots = detect_saed_spots("/path/to/saed_image.png")
    for s in spots:
        print(f"  ({s['cx']:.1f}, {s['cy']:.1f})  radius={s['radius']:.1f}  conf={s['confidence']:.3f}")

首次调用会自动构建 SAM 3 模型（约 6-7 秒），后续调用复用模型。
"""

import os
import sys
import time
import gc
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 0. 修复 fused.py 的 dtype 不匹配问题（在导入模型前执行）
# ---------------------------------------------------------------------------
import sam3.model.vitdet as _vitdet_module

def _patched_addmm_act(activation, linear, mat1):
    x = F.linear(mat1, linear.weight, linear.bias)
    if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
        x = F.relu(x)
    elif activation in [torch.nn.functional.gelu, torch.nn.GELU]:
        x = F.gelu(x)
    return x

_vitdet_module.addmm_act = _patched_addmm_act

# ---------------------------------------------------------------------------
# 1. 全局模型缓存（避免重复构建）
# ---------------------------------------------------------------------------
_model = None
_processor = None
_device = None


def _build_model():
    """构建 SAM 3 模型（仅首次调用时执行）"""
    global _model, _processor, _device

    if _model is not None:
        return _model, _processor, _device

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    # 定位 checkpoint
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _ckpt = os.path.join(_root, "checkpoints", "sam3.pt")

    if not os.path.exists(_ckpt):
        raise FileNotFoundError(
            f"SAM 3 checkpoint not found at {_ckpt}. "
            "Please download it and place it in the checkpoints/ directory."
        )

    _model = build_sam3_image_model(
        checkpoint_path=_ckpt,
        device=_device,
        eval_mode=True,
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )

    _processor = Sam3Processor(
        _model, resolution=1008, device=_device, confidence_threshold=0.0,
    )

    return _model, _processor, _device


def _auto_threshold(scores, skip_top=1):
    """自动分析分数分布，用肘部法确定阈值（跳过透射斑的最高分）"""
    s = np.sort(scores)[::-1]
    s_rest = s[skip_top:]
    if len(s_rest) < 2:
        return 0.0  # 分数太少，不设阈值
    drops = [s_rest[i] - s_rest[i+1] for i in range(min(30, len(s_rest)-1))]
    elbow = np.argmax(drops) + 1
    return float(s_rest[elbow])


def detect_saed_spots(
    image_path: str,
    prompt: str = "a bright spot",
    skip_top: int = 1,
    return_masks: bool = False,
    verbose: bool = False,
) -> dict:
    """
    对 SAED 衍射图像进行斑点检测。

    参数
    ----
    image_path : str
        输入图像路径（PNG/TIFF 等 PIL 支持的格式）。
    prompt : str
        SAM 3 文本提示词，默认 "a bright spot"。
    skip_top : int
        自动阈值时跳过的最高分数量（用于排除透射斑），默认 1。
    return_masks : bool
        是否返回掩码数组（默认 False，节省内存）。
    verbose : bool
        是否打印详细信息。

    返回
    ----
    dict 包含:
        - "spots": list[dict] — 每个斑点的信息
            cx, cy: 质心坐标（像素）
            radius: 等效半径（像素）
            intensity: 积分强度
            peak: 峰值强度
            confidence: SAM 3 置信度
        - "image_size": (width, height) — 输入图像尺寸
        - "threshold": float — 自动确定的阈值
        - "n_spots": int — 检测到的斑点数量
    """
    global _model, _processor, _device
    t0 = time.time()

    # ---- 构建/复用模型 ----
    _model, _processor, _device = _build_model()

    # ---- 加载图像 ----
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    orig_size = img.size

    # 缩放到 SAM 3 内部分辨率（省显存）
    if max(img.size) > 1008:
        img.thumbnail((1008, 1008), Image.LANCZOS)

    if verbose:
        print(f"[SAM3] Loaded {image_path} ({orig_size[0]}x{orig_size[1]})")

    # ---- 推理 ----
    state = _processor.set_image(img)

    # 清 GPU 缓存（避免前一次推理残留）
    if _device == "cuda":
        torch.cuda.empty_cache()

    output = _processor.set_text_prompt(prompt=prompt, state=state)

    scores_np = output["scores"].cpu().numpy()
    masks_tensor = output["masks"]
    boxes_tensor = output["boxes"]

    # ---- 自动阈值 ----
    threshold = _auto_threshold(scores_np, skip_top=skip_top)
    keep = scores_np > threshold

    masks = masks_tensor[keep]
    boxes = boxes_tensor[keep]
    scores = scores_np[keep]

    if verbose:
        elapsed = time.time() - t0
        print(f"[SAM3] {len(scores)}/{len(scores_np)} spots kept "
              f"(threshold={threshold:.4f}, {elapsed:.2f}s)")

    # ---- 提取斑点参数 ----
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    gray = img_np.mean(axis=2) if img_np.ndim == 3 else img_np

    # 转换 masks 到 numpy
    masks_np = masks.cpu().numpy().squeeze()
    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0]

    # 确保 3D (N, H, W)
    if masks_np.ndim == 2:
        masks_np = masks_np[np.newaxis, :, :]

    spots = []
    for idx in range(len(masks_np)):
        mask = masks_np[idx]

        # 如果尺寸不匹配，缩放到图像尺寸
        if mask.shape != (h, w):
            from skimage.transform import resize
            mask = resize(mask.astype(float), (h, w), preserve_range=True) > 0.5

        if np.sum(mask) == 0:
            continue

        ys, xs = np.where(mask)
        spot = {
            "cx": float(np.mean(xs)),
            "cy": float(np.mean(ys)),
            "radius": float(np.sqrt(np.sum(mask) / np.pi)),
            "intensity": float(np.sum(gray[mask])),
            "peak": float(np.max(gray[mask])),
            "confidence": float(scores[idx]) if idx < len(scores) else 0.0,
        }
        spot["dist_to_center"] = float(
            np.sqrt((spot["cx"] - w / 2) ** 2 + (spot["cy"] - h / 2) ** 2)
        )
        spots.append(spot)

    # 按距中心距离排序
    spots.sort(key=lambda s: s["dist_to_center"])

    result = {
        "spots": spots,
        "image_size": orig_size,
        "threshold": threshold,
        "n_spots": len(spots),
    }

    if return_masks:
        result["masks"] = masks_np

    # 清理
    del state, output, masks_tensor, boxes_tensor
    if _device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return result


def batch_detect(
    image_paths: list,
    prompt: str = "a bright spot",
    verbose: bool = True,
) -> list:
    """
    批量检测多张 SAED 图像。

    参数
    ----
    image_paths : list[str]
        图像路径列表。
    prompt : str
        文本提示词。
    verbose : bool
        是否打印进度。

    返回
    ----
    list[dict] — 每张图对应一个 detect_saed_spots() 的返回值。
    """
    results = []
    for i, path in enumerate(image_paths):
        if verbose:
            print(f"[{i + 1}/{len(image_paths)}] {os.path.basename(path)}")
        res = detect_saed_spots(path, prompt=prompt, verbose=False)
        results.append(res)
        if verbose:
            print(f"       -> {res['n_spots']} spots")
    return results


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAED diffraction spot detection with SAM 3")
    parser.add_argument("image", help="Path to SAED image (PNG/TIFF)")
    parser.add_argument("--prompt", default="a bright spot", help="SAM 3 text prompt")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print details")

    args = parser.parse_args()

    result = detect_saed_spots(
        args.image,
        prompt=args.prompt,
        verbose=args.verbose,
    )

    print(f"\nDetected {result['n_spots']} diffraction spots "
          f"(threshold={result['threshold']:.3f}):")
    print(f"{'#':>3} | {'Cx':>7} | {'Cy':>7} | {'Radius':>5} | {'Intensity':>9} | {'Conf':>6}")
    print("-" * 45)
    for i, s in enumerate(result["spots"]):
        print(f"{i + 1:>3} | {s['cx']:>7.1f} | {s['cy']:>7.1f} | "
              f"{s['radius']:>5.1f} | {s['intensity']:>9.0f} | {s['confidence']:>6.3f}")
