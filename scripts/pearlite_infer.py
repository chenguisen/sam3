#!/usr/bin/env python3
"""
珠光体 (Pearlite) 自动检测与标注脚本
===================================
使用 SAM 3 模型从 TEM 图像中自动检测珠光体区域，绘制边界框并保存结果。

工作流程:
  1. 自动加载 SAM 3 模型 + checkpoint
  2. 对 data/3/ 中的每张图片用文本提示词推理
  3. 在特征区域绘制边界框
  4. 交互式查看和微调（添加/删除框）
  5. 保存标注为 JSON（兼容 labelme 格式）

用法:
  # 使用项目 venv 运行
  .venv/bin/python scripts/pearlite_infer.py

  # 指定图片目录和提示词
  .venv/bin/python scripts/pearlite_infer.py --data_dir data/3 --prompt "pearlite colony"

  # 只推理不显示 GUI（批量模式）
  .venv/bin/python scripts/pearlite_infer.py --headless

依赖: torch, numpy, Pillow, matplotlib, skimage
"""

import sys
import os
import json
import glob
import gc
import time
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# 0. 修复 fused.py 的 dtype 不匹配问题（在导入模型前执行）
# ---------------------------------------------------------------------------
import sam3.model.vitdet as _vitdet_module


def _patched_addmm_act(activation, linear, mat1):
    import torch.nn.functional as F

    x = F.linear(mat1, linear.weight, linear.bias)
    if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
        x = F.relu(x)
    elif activation in [torch.nn.functional.gelu, torch.nn.GELU]:
        x = F.gelu(x)
    return x


_vitdet_module.addmm_act = _patched_addmm_act

# 设置 matplotlib 后端（在导入 pyplot 前）
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle, Polygon
from matplotlib.widgets import Button

# 中文字体设置
_zh_font_set = False
for _f in fm.findSystemFonts():
    if "NotoSansCJK" in _f or "NotoSerifCJK" in _f or "WenQuanYi" in _f or "SimHei" in _f:
        plt.rcParams["font.family"] = fm.FontProperties(fname=_f).get_name()
        _zh_font_set = True
        break
if not _zh_font_set:
    # 尝试设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import torch
import torch.nn.functional as F

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ---------------------------------------------------------------------------
# 项目路径常量
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "3"
DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / "sam3.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "3" / "annotations"

# 珠光体候选提示词（按推荐优先级排序）
# TEM 图像中的珠光体在灰度图上表现为深色区域，提示词越具体越容易命中
PEARLITE_PROMPTS = [
    "dark irregular region in metal",
    "dark area in steel microstructure",
    "dark etched region at grain boundary",
    "small dark object",
    "dark spot",
    "dark region in grayscale image",
    "pearlite colony in steel",
    "dark lamellar structure",
    "pearlite",
    "pearlite colony",
    "dark phase",
]

# 颜色循环（用于绘制不同检测结果）
COLORS = plt.cm.tab10
MASK_COLORS_RGB = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
    (0, 128, 255), (255, 64, 128), (128, 255, 0), (64, 128, 255),
]


# ===================================================================
#  核心检测函数
# ===================================================================

class PearliteDetector:
    """珠光体检测器：封装 SAM 3 模型，支持文本提示词检测和框提示检测。"""

    def __init__(self, checkpoint_path=None, device=None, confidence=0.3):
        if checkpoint_path is None:
            checkpoint_path = str(DEFAULT_CKPT)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.confidence = confidence

        print(f"🔄 加载 SAM 3 模型...", end=" ", flush=True)
        t0 = time.time()
        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            device=device,
            eval_mode=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )
        self.processor = Sam3Processor(
            self.model, resolution=1008, device=device,
            confidence_threshold=confidence,
        )
        print(f"✅ 完成 ({time.time() - t0:.1f}s)")

    @torch.inference_mode()
    def detect_by_text(self, image, prompt="pearlite colony in steel"):
        """
        用文本提示词检测珠光体区域。

        参数:
            image: PIL.Image 或 numpy 数组
            prompt: 文本提示词

        返回: dict 包含:
            - "boxes": list of [x1, y1, x2, y2] 绝对坐标
            - "scores": list of float 置信度
            - "masks": list of (H, W) bool 掩码
            - "image_size": (W, H)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        orig_w, orig_h = image.size

        # 预处理：缩放到 SAM3 内部分辨率
        img_resized = image.copy()
        if max(img_resized.size) > 1008:
            img_resized.thumbnail((1008, 1008), Image.LANCZOS)

        state = self.processor.set_image(img_resized)
        if self.device == "cuda":
            torch.cuda.empty_cache()

        state = self.processor.set_text_prompt(prompt=prompt, state=state)

        scores = state["scores"].cpu().numpy()
        masks = state["masks"].cpu().numpy()  # (N, 1, H, W)
        boxes = state["boxes"].cpu().numpy()  # (N, 4) in [x1,y1,x2,y2] absolute

        # 过滤低置信度
        keep = scores > self.confidence
        scores = scores[keep].tolist()
        masks = masks[keep]
        boxes = boxes[keep]

        # 将 masks / boxes 从缩略图坐标映射回原图坐标
        scale_x = orig_w / img_resized.size[0]
        scale_y = orig_h / img_resized.size[1]

        boxes_orig = []
        masks_orig = []
        for i in range(len(boxes)):
            b = boxes[i]
            boxes_orig.append([
                float(b[0] * scale_x),
                float(b[1] * scale_y),
                float(b[2] * scale_x),
                float(b[3] * scale_y),
            ])
            m = masks[i, 0]  # (H_resized, W_resized)
            if m.shape != (orig_h, orig_w):
                from skimage.transform import resize
                m = resize(m.astype(float), (orig_h, orig_w),
                           preserve_range=True) > 0.5
            masks_orig.append(m)

        # 清理
        del state
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return {
            "boxes": boxes_orig,
            "scores": scores,
            "masks": masks_orig,
            "image_size": (orig_w, orig_h),
            "n_detections": len(boxes_orig),
        }

    @torch.inference_mode()
    def detect_by_box(self, image, boxes_normalized, prompt="visual"):
        """
        用边界框提示在指定区域检测。
        TEM 图像推荐使用 prompt="visual"（纯几何提示，不依赖文本理解）。

        参数:
            image: PIL.Image
            boxes_normalized: list of [cx, cy, w, h] 归一化到 [0,1]
            prompt: 文本提示词。设为 "visual" 可关闭文本分支，只靠框推理。

        返回: dict 同上
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        orig_w, orig_h = image.size

        img_resized = image.copy()
        if max(img_resized.size) > 1008:
            img_resized.thumbnail((1008, 1008), Image.LANCZOS)

        state = self.processor.set_image(img_resized)

        # 设置文本（"visual" 表示仅视觉，不依赖文本语义）
        text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
        state["backbone_out"].update(text_outputs)
        state["geometric_prompt"] = self.model._get_dummy_prompt()

        # 添加所有框
        for box in boxes_normalized:
            b = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
            lbl = torch.tensor([True], device=self.device, dtype=torch.bool).view(1, 1)
            state["geometric_prompt"].append_boxes(b, lbl)

        # 推理
        state = self.processor._forward_grounding(state)

        scores = state["scores"].cpu().numpy()
        masks = state["masks"].cpu().numpy()
        boxes = state["boxes"].cpu().numpy()

        keep = scores > self.confidence
        scores = scores[keep].tolist()
        masks = masks[keep]
        boxes = boxes[keep]

        scale_x = orig_w / img_resized.size[0]
        scale_y = orig_h / img_resized.size[1]

        boxes_orig = []
        masks_orig = []
        for i in range(len(boxes)):
            b = boxes[i]
            boxes_orig.append([
                float(b[0] * scale_x), float(b[1] * scale_y),
                float(b[2] * scale_x), float(b[3] * scale_y),
            ])
            m = masks[i, 0]
            if m.shape != (orig_h, orig_w):
                from skimage.transform import resize
                m = resize(m.astype(float), (orig_h, orig_w),
                           preserve_range=True) > 0.5
            masks_orig.append(m)

        del state
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return {
            "boxes": boxes_orig,
            "scores": scores,
            "masks": masks_orig,
            "image_size": (orig_w, orig_h),
            "n_detections": len(boxes_orig),
        }

    # ------------------------------------------------------------------
    # TEM 图像专用：自动候选区域检测
    # 注意: TEM 图像与自然图像差异大，自动候选仅辅助，推荐使用交互式画框。
    # ------------------------------------------------------------------
    def find_candidate_regions(self, image, min_area=50, max_area=50000,
                                contrast_threshold=30, blur_sigma=3.0):
        """
        用图像处理方法自动找出 TEM 图像中的深色候选区域（珠光体候选）。
        返回归一化框坐标列表 [cx, cy, w, h]。

        参数:
            image: PIL.Image 或 numpy 数组 (RGB 或灰度)
            min_area: 最小区域面积（像素）
            max_area: 最大区域面积（像素）
            contrast_threshold: 自适应阈值灵敏度 (越小越敏感)
            blur_sigma: 高斯模糊标准差 (去噪)
        """
        from skimage import filters, morphology, measure
        from scipy import ndimage as ndi

        if isinstance(image, Image.Image):
            img_np = np.array(image.convert("L"))
        else:
            if image.ndim == 3:
                img_np = np.mean(image, axis=2).astype(np.uint8)
            else:
                img_np = image.astype(np.uint8)

        h, w = img_np.shape

        # 1. 高斯模糊去噪
        blurred = filters.gaussian(img_np, sigma=blur_sigma)

        # 2. OTSU 全局阈值找深色区域
        try:
            threshold = filters.threshold_otsu(blurred)
            # 稍微严格一点（珠光体比平均暗）
            threshold = threshold * 0.85
        except Exception:
            threshold = 0.3  # fallback

        # 3. 二值化
        binary = blurred < threshold

        # 4. 形态学操作去除噪声
        # 注意: 新版本 skimage 使用 max_size 替代 min_size
        try:
            binary = morphology.remove_small_objects(binary, max_size=min_area)
            binary = morphology.remove_small_holes(binary, max_size=min_area)
        except TypeError:
            # 兼容旧版本
            binary = morphology.remove_small_objects(binary, min_size=min_area)
            binary = morphology.remove_small_holes(binary, area_threshold=min_area)
        binary = ndi.binary_fill_holes(binary)

        # 5. 连通域标记
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)

        boxes = []
        for region in regions:
            area = region.area
            if area < min_area or area > max_area:
                continue
            minr, minc, maxr, maxc = region.bbox
            # 过滤掉太靠近边缘的区域（通常是边界伪影）
            margin = 20
            if minr < margin or minc < margin or maxr > h - margin or maxc > w - margin:
                continue
            # 过滤掉细长条（通常是划痕/晶界，不是珠光体）
            box_h = maxr - minr
            box_w = maxc - minc
            aspect_ratio = max(box_h, box_w) / (min(box_h, box_w) + 1)
            if aspect_ratio > 5:
                continue
            # 计算归一化框
            box_cx = (minc + maxc) / 2 / w
            box_cy = (minr + maxr) / 2 / h
            box_w_norm = (maxc - minc) / w
            box_h_norm = (maxr - minr) / h
            boxes.append([box_cx, box_cy, box_w_norm, box_h_norm])

        return boxes

    def detect_tem_auto(self, image, prompt="visual", min_area=50, max_area=50000,
                        contrast_threshold=30, blur_sigma=3.0):
        """
        TEM 图像全自动珠光体检测流程：
          1. 图像处理找候选深色区域
          2. 用 SAM3 框提示分割每个候选区域
          3. 返回分割结果

        参数:
            image: PIL.Image
            prompt: 传给 SAM3 的文本提示
            min_area, max_area: 候选区域面积范围
            contrast_threshold, blur_sigma: 图像预处理参数

        返回: dict (同 detect_by_text)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # 步骤 1: 找候选区域
        candidates = self.find_candidate_regions(
            image, min_area=min_area, max_area=max_area,
            contrast_threshold=contrast_threshold, blur_sigma=blur_sigma,
        )

        if not candidates:
            return {
                "boxes": [], "scores": [], "masks": [],
                "image_size": image.size, "n_detections": 0,
            }

        # 步骤 2: 用 SAM3 分割
        return self.detect_by_box(image, candidates, prompt=prompt)


# ===================================================================
#  可视化与交互
# ===================================================================

class PearliteViewer:
    """交互式查看珠光体检测结果。"""

    def __init__(self, detector, data_dir, output_dir, prompt="pearlite colony in steel",
                 auto_mode=True):
        self.detector = detector
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt = prompt
        self.auto_mode = auto_mode  # True=自动候选检测, False=纯文本提示

        # 扫描图片
        self.image_paths = sorted([
            p for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")
            for p in glob.glob(str(self.data_dir / ext))
        ])
        if not self.image_paths:
            print(f"❌ {data_dir} 中未找到图片")
            sys.exit(1)
        print(f"📷 找到 {len(self.image_paths)} 张图片")

        # 每张图的结果缓存
        self.results = [None] * len(self.image_paths)

        # 当前状态
        self.idx = 0
        self.current_img = None
        self.fig = None
        self.ax_img = None
        self.ax_status = None
        self.img_display = None
        self.overlay = None
        self.box_patches = []
        self.mask_overlay = None
        self.status_text = None

        # 编辑状态
        self.drawing = False
        self.draw_start = None
        self.draw_rect = None
        self.removing = False

    def run(self):
        """启动交互式查看器。"""
        print(f"\n🔍 提示词: '{self.prompt}'  模式: {'自动候选' if self.auto_mode else '纯文本'}")
        print("🖱️  操作说明:")
        print("  左键拖动: 画框标注 | 点击框内: 删除")
        print("  滚轮: 缩放  |  中键拖动: 平移  |  右键拖动: 画框")
        print("  [z] 重置缩放  [n/p] 翻页  [r] 重推理  [a] 切换模式")
        print("  [s] 保存  [q] 退出\n")

        self._setup_gui()
        self._load_and_show(self.idx)
        plt.show()

    def _setup_gui(self):
        self.fig, (self.ax_img, self.ax_status) = plt.subplots(
            2, 1, figsize=(14, 11),
            gridspec_kw={"height_ratios": [10, 1]},
        )
        self.fig.canvas.manager.set_window_title("珠光体检测 - Pearlite Detector")
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.ax_status.axis("off")

        # 缩放/平移状态
        self.zoom_level = 1.0
        self.pan_start = None
        self.panning = False

    def _load_and_show(self, idx):
        """加载指定索引的图片并显示。"""
        self.idx = idx
        path = self.image_paths[idx]
        img = np.array(Image.open(path).convert("RGB"))
        self.current_img = img

        # 如果还没有推理结果，自动推理
        if self.results[idx] is None:
            self._run_inference(idx, auto=self.auto_mode)

        self._redraw()

    def _run_inference(self, idx, auto=True):
        """对指定图片运行推理。

        参数:
            idx: 图片索引
            auto: True=用图像处理自动找候选区域; False=仅文本提示词
        """
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        print(f"  ⏳ 推理 {Path(path).name} ...", end=" ", flush=True)
        t0 = time.time()

        if auto and False:  # 默认先用纯文本，后续可切换
            result = self.detector.detect_by_text(img, prompt=self.prompt)
        else:
            # 先用图像处理找候选，再用 SAM3 框提示分割
            result = self.detector.detect_tem_auto(
                img, prompt="visual",
                min_area=30, max_area=50000,
            )
            # 如果没有自动检测到，尝试文本提示
            if result["n_detections"] == 0:
                print(f"(自动候选未找到，尝试文本提示) ", end="")
                result = self.detector.detect_by_text(img, prompt=self.prompt)

        self.results[idx] = result
        print(f"检测到 {result['n_detections']} 个区域 ({time.time() - t0:.1f}s)")

    def _redraw(self):
        """重新绘制当前图像和标注，保持缩放状态。"""
        # 保存当前视图范围
        xlim = self.ax_img.get_xlim()
        ylim = self.ax_img.get_ylim()
        was_zoomed = not (xlim[0] == 0 and ylim[0] == 0
                          and abs(xlim[1] - self.current_img.shape[1]) < 1
                          and abs(ylim[1] - self.current_img.shape[0]) < 1)

        self.ax_img.clear()
        result = self.results[self.idx]
        img = self.current_img
        h, w = img.shape[:2]

        # 显示原图
        self.ax_img.imshow(img)
        self.ax_img.set_title(
            f"[{self.idx + 1}/{len(self.image_paths)}] "
            f"{Path(self.image_paths[self.idx]).name}  |  "
            f"检测到 {result['n_detections']} 个珠光体区域",
            fontsize=12,
        )
        self.ax_img.axis("on")

        # 绘制掩码（半透明叠加）
        if result["masks"] and result["n_detections"] > 0:
            overlay = np.zeros((h, w, 4), dtype=np.float32)
            for i, mask in enumerate(result["masks"]):
                color = COLORS(i % 10)[:3]
                for c in range(3):
                    overlay[mask, c] = overlay[mask, c] * 0.7 + color[c] * 0.3
                overlay[mask, 3] = np.maximum(overlay[mask, 3], 0.4)
            overlay[:, :, 3] = np.clip(overlay[:, :, 3], 0, 1)
            self.ax_img.imshow(overlay)

        # 绘制边界框
        self.box_patches.clear()
        for i, box in enumerate(result["boxes"]):
            x1, y1, x2, y2 = box
            color = COLORS(i % 10)
            rect = Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor=color, linewidth=2, alpha=0.9,
            )
            self.ax_img.add_patch(rect)
            self.box_patches.append(rect)

            # 显示置信度
            score = result["scores"][i]
            self.ax_img.text(
                x1, y1 - 3, f"{score:.2f}",
                fontsize=8, color=color, weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

        # 恢复缩放状态
        if was_zoomed:
            self.ax_img.set_xlim(xlim)
            self.ax_img.set_ylim(ylim)
        else:
            # imshow 默认 origin='upper'，保持默认即可
            self.ax_img.set_xlim(0, w)
            self.zoom_level = 1.0

        # 状态栏
        self.ax_status.clear()
        self.ax_status.axis("off")
        n = result["n_detections"]
        mode_str = "自动候选" if self.auto_mode else "文本提示"
        status = (
            f"📷 {Path(self.image_paths[self.idx]).name}  "
            f"| 🎯 {n} 个珠光体区域  "
            f"| 🔤 模式: {mode_str}  "
            f"| [n/p] 翻页  [a] 切换模式  [r] 重推理  [s] 保存  [q] 退出"
        )
        self.ax_status.text(0.5, 0.5, status, ha="center", va="center",
                            fontsize=10, transform=self.ax_status.transAxes,
                            bbox=dict(boxstyle="round", facecolor="lightyellow"))

        self.fig.canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self.ax_img:
            return
        x, y = event.xdata, event.ydata

        # 中键：平移
        if event.button == 2:
            self.panning = True
            self.pan_start = (x, y)
            return

        # 左键/右键：画框 或 删除已有框
        if event.button not in (1, 3):
            return

        # 检查是否点击了已有框（删除模式）
        result = self.results[self.idx]
        for i in range(len(result["boxes"]) - 1, -1, -1):
            x1, y1, x2, y2 = result["boxes"][i]
            if x1 <= x <= x2 and y1 <= y <= y2:
                # 删除该检测
                result["boxes"].pop(i)
                result["scores"].pop(i)
                result["masks"].pop(i)
                result["n_detections"] = len(result["boxes"])
                print(f"  🗑️ 删除检测 #{i + 1}")
                self._redraw()
                return

        # 开始画新框
        self.drawing = True
        self.draw_start = (x, y)
        self.draw_rect = Rectangle(
            (x, y), 0, 0, fill=False, edgecolor="cyan",
            linewidth=2, linestyle="--",
        )
        self.ax_img.add_patch(self.draw_rect)

    def _on_move(self, event):
        if event.xdata is None or event.ydata is None:
            return

        # 平移中
        if self.panning and self.pan_start:
            x0, y0 = self.pan_start
            dx, dy = event.xdata - x0, event.ydata - y0
            xlim = self.ax_img.get_xlim()
            ylim = self.ax_img.get_ylim()
            self.ax_img.set_xlim(xlim[0] - dx, xlim[1] - dx)
            self.ax_img.set_ylim(ylim[0] - dy, ylim[1] - dy)
            self.pan_start = (event.xdata, event.ydata)
            self.fig.canvas.draw_idle()
            return

        # 画框中
        if not self.drawing or not self.draw_rect:
            return
        x0, y0 = self.draw_start
        x, y = event.xdata, event.ydata
        self.draw_rect.set_xy((min(x0, x), min(y0, y)))
        self.draw_rect.set_width(abs(x - x0))
        self.draw_rect.set_height(abs(y - y0))
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        # 结束平移
        if self.panning:
            self.panning = False
            self.pan_start = None
            return

        if not self.drawing or not self.draw_rect:
            return
        self.drawing = False
        x0, y0 = self.draw_start
        x, y = event.xdata, event.ydata
        w, h = abs(x - x0), abs(y - y0)

        if w > 10 and h > 10:
            # 用户画了新框，用框提示进行局部推理
            x1, y1 = min(x0, x), min(y0, y)
            x2, y2 = max(x0, x), max(y0, y)
            img_w, img_h = self.current_img.shape[1], self.current_img.shape[0]

            # 归一化框坐标 (cx, cy, w, h)
            box_cx = (x1 + x2) / 2 / img_w
            box_cy = (y1 + y2) / 2 / img_h
            box_w = (x2 - x1) / img_w
            box_h = (y2 - y1) / img_h

            print(f"  📦 框提示推理 ({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)}) ...",
                  end=" ", flush=True)
            t0 = time.time()
            local_result = self.detector.detect_by_box(
                Image.fromarray(self.current_img),
                boxes_normalized=[[box_cx, box_cy, box_w, box_h]],
                prompt=self.prompt,
            )
            print(f"检测到 {local_result['n_detections']} 个 ({time.time() - t0:.1f}s)")

            # 合并结果
            result = self.results[self.idx]
            result["boxes"].extend(local_result["boxes"])
            result["scores"].extend(local_result["scores"])
            result["masks"].extend(local_result["masks"])
            result["n_detections"] = len(result["boxes"])

        if self.draw_rect in self.ax_img.patches:
            self.draw_rect.remove()
        self.draw_rect = None
        self._redraw()

    def _on_scroll(self, event):
        """滚轮缩放，以鼠标位置为中心。"""
        if event.inaxes != self.ax_img:
            return

        scale_factor = 1.15
        if event.button == "up":
            self.zoom_level *= scale_factor
        elif event.button == "down":
            self.zoom_level /= scale_factor
        else:
            return

        # 限制缩放范围
        h, w = self.current_img.shape[:2]
        self.zoom_level = max(0.1, min(self.zoom_level, 50.0))

        # 以鼠标位置为中心缩放
        x, y = event.xdata, event.ydata
        xlim = self.ax_img.get_xlim()
        ylim = self.ax_img.get_ylim()

        # 计算缩放因子（相对于当前视图）
        if event.button == "up":
            cur_scale = 1.0 / scale_factor
        else:
            cur_scale = scale_factor

        new_width = (xlim[1] - xlim[0]) * cur_scale
        new_height = (ylim[1] - ylim[0]) * cur_scale

        self.ax_img.set_xlim(x - new_width * (x - xlim[0]) / (xlim[1] - xlim[0]),
                             x + new_width * (xlim[1] - x) / (xlim[1] - xlim[0]))
        self.ax_img.set_ylim(y - new_height * (y - ylim[0]) / (ylim[1] - ylim[0]),
                             y + new_height * (ylim[1] - y) / (ylim[1] - ylim[0]))

        self.fig.canvas.draw_idle()

    def _reset_view(self):
        """重置缩放到完整图像。"""
        if self.current_img is None:
            return
        h, w = self.current_img.shape[:2]
        self.ax_img.set_xlim(0, w)
        self.ax_img.set_ylim(h, 0)
        self.zoom_level = 1.0
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == "z" or event.key == "Z":
            self._reset_view()
        elif event.key == "n":
            self._save_current()
            if self.idx < len(self.image_paths) - 1:
                self._load_and_show(self.idx + 1)
        elif event.key == "p":
            if self.idx > 0:
                self._load_and_show(self.idx - 1)
        elif event.key == "a":
            # 切换模式
            self.auto_mode = not self.auto_mode
            mode_name = "自动候选检测" if self.auto_mode else "纯文本提示"
            print(f"  🔄 切换至 {mode_name} 模式")
            self.results[self.idx] = None
            self._load_and_show(self.idx)
        elif event.key == "r":
            # 重新推理
            self.results[self.idx] = None
            self._load_and_show(self.idx)
        elif event.key == "s":
            self._save_current()
        elif event.key == "q":
            self._quit()

    def _save_current(self):
        """保存当前图片的标注结果。"""
        idx = self.idx
        result = self.results[idx]
        if result is None or result["n_detections"] == 0:
            print("  ⚠️  无检测结果，跳过保存")
            return

        path = self.image_paths[idx]
        name = Path(path).name
        stem = Path(path).stem
        h, w = self.current_img.shape[:2]

        # 计算总面积占比
        total_area = h * w
        mask_area = 0
        if result["masks"]:
            mask_area = int(sum(m.sum() for m in result["masks"]))

        # 保存为注释 json（兼容 labelme 格式）
        shapes = []
        for i, box in enumerate(result["boxes"]):
            x1, y1, x2, y2 = box
            shapes.append({
                "label": "珠光体",
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "description": f"confidence={result['scores'][i]:.3f}",
                "shape_type": "rectangle",
                "flags": {},
            })

        data = {
            "version": "1.0",
            "image": name,
            "image_size": [w, h],
            "n_detections": result["n_detections"],
            "area_total_pixels": total_area,
            "area_annotated_pixels": int(mask_area),
            "area_percentage": round(mask_area / total_area * 100, 4) if total_area else 0,
            "prompt": self.prompt,
            "boxes": [
                {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
                 "score": result["scores"][i]}
                for i, b in enumerate(result["boxes"])
            ],
            "shapes": shapes,
        }

        save_path = self.output_dir / f"{stem}.json"
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  ✅ 已保存: {save_path}  (珠光体面积占比 {data['area_percentage']:.2f}%)")

    def _quit(self):
        self._save_current()
        plt.close("all")
        print("👋 退出")
        sys.exit(0)


# ===================================================================
#  批量推理（无 GUI）
# ===================================================================

def batch_inference(detector, data_dir, output_dir, prompt="pearlite colony in steel",
                    use_auto=True):
    """无交互的批量推理。

    参数:
        use_auto: True=用图像处理自动找候选; False=仅文本提示
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")
        for p in glob.glob(str(data_dir / ext))
    ])

    if not image_paths:
        print(f"❌ {data_dir} 中未找到图片")
        return

    mode_str = "自动候选检测" if use_auto else "文本提示"
    print(f"\n{'='*60}")
    print(f"批量推理模式")
    print(f"  图片目录: {data_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  检测模式: {mode_str}")
    print(f"  图片数:   {len(image_paths)}")
    print(f"{'='*60}\n")

    total_time = 0
    for i, path in enumerate(image_paths):
        name = Path(path).name
        print(f"[{i + 1}/{len(image_paths)}] {name} ...", end=" ", flush=True)

        t0 = time.time()
        img = Image.open(path).convert("RGB")
        if use_auto:
            result = detector.detect_tem_auto(
                img, prompt="visual",
                min_area=30, max_area=50000,
            )
            if result["n_detections"] == 0:
                result = detector.detect_by_text(img, prompt=prompt)
        else:
            result = detector.detect_by_text(img, prompt=prompt)
        elapsed = time.time() - t0
        total_time += elapsed

        # 保存结果
        h, w = result["image_size"][1], result["image_size"][0]
        mask_area = int(sum(m.sum() for m in result["masks"])) if result["masks"] else 0

        data = {
            "image": name,
            "image_size": [w, h],
            "n_detections": result["n_detections"],
            "area_total_pixels": w * h,
            "area_annotated_pixels": mask_area,
            "area_percentage": round(mask_area / (w * h) * 100, 4) if w * h else 0,
            "prompt": prompt,
            "boxes": [
                {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
                 "score": result["scores"][i]}
                for i, b in enumerate(result["boxes"])
            ],
        }

        save_path = output_dir / f"{Path(path).stem}.json"
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"检测到 {result['n_detections']} 个  |  "
              f"面积占比 {data['area_percentage']:.2f}%  |  "
              f"{elapsed:.1f}s")

    avg = total_time / len(image_paths)
    print(f"\n📊 总计: {len(image_paths)} 张, {total_time:.1f}s (平均 {avg:.1f}s/张)")


# ===================================================================
#  测试不同提示词
# ===================================================================

def test_prompts(detector, data_dir):
    """测试多个提示词，找出效果最好的。同时测试自动候选检测模式。"""
    data_dir = Path(data_dir)
    image_paths = sorted([
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")
        for p in glob.glob(str(data_dir / ext))
    ])
    if not image_paths:
        print(f"❌ {data_dir} 中未找到图片")
        return

    # 只用第一张图测试
    test_img = Image.open(image_paths[0]).convert("RGB")
    w, h = test_img.size
    print(f"\n📷 测试图片: {Path(image_paths[0]).name} ({w}x{h})")
    print(f"{'='*60}")

    # 先测自动候选模式
    print(f"\n🔍 自动候选检测模式:")
    t0 = time.time()
    auto_result = detector.detect_tem_auto(
        test_img, prompt="visual",
        min_area=30, max_area=50000,
    )
    auto_time = time.time() - t0
    print(f"  {'自动候选检测 (SAM3 框提示)':<40} | {auto_result['n_detections']:>6} | {auto_time:>5.1f}s")

    # 再测文本提示词
    print(f"\n🔍 文本提示词模式:")
    print(f"{'提示词':<40} | {'检测数':>6} | {'时间':>6}")
    print(f"{'-'*60}")

    results = []
    for prompt in PEARLITE_PROMPTS:
        t0 = time.time()
        result = detector.detect_by_text(test_img, prompt=prompt)
        elapsed = time.time() - t0
        print(f"{prompt:<40} | {result['n_detections']:>6} | {elapsed:>5.1f}s")
        results.append((prompt, result))

    # 推荐最佳方法
    best_method = "自动候选检测"
    best_count = auto_result["n_detections"]
    valid = [(p, r) for p, r in results if r["n_detections"] > best_count]
    if valid:
        best = max(valid, key=lambda x: x[1]["n_detections"])
        best_method = f"文本提示词 '{best[0]}'"
        best_count = best[1]["n_detections"]

    if best_count > 0:
        print(f"\n💡 推荐方法: {best_method} (检测到 {best_count} 个区域)")
    else:
        print(f"\n⚠️  所有方法均未检测到结果。建议:")
        print(f"   1. 使用交互模式 (.venv/bin/python scripts/pearlite_infer.py)")
        print(f"   2. 手动画框标出珠光体区域")
        print(f"   3. 按 [s] 保存标注，作为后续训练的参考")

    return results


# ===================================================================
#  入口
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="珠光体 (Pearlite) 自动检测与标注 — SAM 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互模式（默认，自动候选检测）
  .venv/bin/python scripts/pearlite_infer.py

  # 交互模式（纯文本提示）
  .venv/bin/python scripts/pearlite_infer.py --no-auto

  # 批量推理（无 GUI）
  .venv/bin/python scripts/pearlite_infer.py --headless

  # 测试不同检测方法
  .venv/bin/python scripts/pearlite_infer.py --test

  # 指定自定义提示词
  .venv/bin/python scripts/pearlite_infer.py --prompt "dark lamellar structure"

  # 指定图片目录
  .venv/bin/python scripts/pearlite_infer.py --data_dir /path/to/images
        """,
    )
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR),
                        help=f"图片目录 (默认: {DEFAULT_DATA_DIR})")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR),
                        help=f"输出目录 (默认: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT),
                        help=f"模型 checkpoint (默认: {DEFAULT_CKPT})")
    parser.add_argument("--prompt", default="pearlite colony in steel",
                        help="文本提示词 (默认: 'pearlite colony in steel')")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="置信度阈值 (默认: 0.3)")
    parser.add_argument("--device", default=None,
                        help="计算设备 (默认: cuda if available else cpu)")
    parser.add_argument("--headless", action="store_true",
                        help="无交互的批量推理模式")
    parser.add_argument("--auto", dest="auto_mode", action="store_true", default=True,
                        help="自动候选检测（默认，图像处理+SAM3框提示）")
    parser.add_argument("--no-auto", dest="auto_mode", action="store_false",
                        help="纯文本提示模式（不使用图像处理候选检测）")
    parser.add_argument("--test", action="store_true",
                        help="测试多个提示词效果")
    args = parser.parse_args()

    # 确保在项目根目录运行
    os.chdir(str(PROJECT_ROOT))

    # 检查 checkpoint
    if not os.path.exists(args.ckpt):
        print(f"❌ 找不到 checkpoint: {args.ckpt}")
        print("   请下载 sam3.pt 放到 checkpoints/ 目录下，或使用 --ckpt 指定路径")
        sys.exit(1)

    # 构建检测器
    detector = PearliteDetector(
        checkpoint_path=args.ckpt,
        device=args.device,
        confidence=args.confidence,
    )

    if args.test:
        # 测试提示词模式
        test_prompts(detector, args.data_dir)
    elif args.headless:
        # 批量推理模式
        batch_inference(detector, args.data_dir, args.output_dir, prompt=args.prompt,
                        use_auto=args.auto_mode)
    else:
        # 交互模式
        viewer = PearliteViewer(
            detector, args.data_dir, args.output_dir, prompt=args.prompt,
            auto_mode=args.auto_mode,
        )
        viewer.run()


if __name__ == "__main__":
    main()
