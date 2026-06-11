#!/usr/bin/env python3
"""
交互式标注 + SAM 3 自动分割工具
================================
工作流:
  1. 手动在图像上画框标注目标区域（如珠光体、晶界等）
  2. SAM 3 自动分割框内目标
  3. 手动微调（添加/删除框）
  4. 保存标注 + 面积占比
  5. 可将标注作为参考模板，批量应用到新图片

快捷键:
  - 左键拖动: 画框标注一个目标
  - 左键点击框内: 删除该框
  - n/p: 下一张/上一张
  - s: 保存当前标注
  - q: 退出

用法:
  python scripts/interactive_annotate.py
  python scripts/interactive_annotate.py --data_dir /path/to/images
"""

import sys, os, json, glob, argparse, gc
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# 中文字体设置
for _f in fm.findSystemFonts():
    if "NotoSansCJK" in _f or "NotoSerifCJK" in _f:
        plt.rcParams["font.family"] = fm.FontProperties(fname=_f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.patches import Rectangle


SAM3_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SAM3_ROOT)

import torch
import torch.nn.functional as F

# ----- 修复 fused.py dtype 问题 -----
import sam3.model.vitdet as _vd
def _fix_addmm(act, lin, m1):
    x = F.linear(m1, lin.weight, lin.bias)
    if act in [torch.nn.functional.relu, torch.nn.ReLU]: x = F.relu(x)
    elif act in [torch.nn.functional.gelu, torch.nn.GELU]: x = F.gelu(x)
    return x
_vd.addmm_act = _fix_addmm

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class Annotator:
    SUPPORTED = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

    def __init__(self, data_dir, output_dir=None):
        self.data_dir = data_dir
        self.output_dir = output_dir or os.path.join(data_dir, "annotations")
        os.makedirs(self.output_dir, exist_ok=True)

        # 扫描图片
        self.paths = sorted(set(
            p for ext in self.SUPPORTED
            for p in glob.glob(os.path.join(data_dir, f"*{ext}"))
            + glob.glob(os.path.join(data_dir, f"*{ext.upper()}"))
        ))
        if not self.paths:
            print(f"❌ {data_dir} 中未找到图片"); sys.exit(1)
        print(f"找到 {len(self.paths)} 张图片")

        self.idx = 0
        self.boxes = [[] for _ in self.paths]  # 每张图的框列表
        self.masks = [None] * len(self.paths)   # 缓存分割结果

        # 交互状态
        self.fig = self.ax = None
        self.img = None
        self.rects = []
        self.rect_patches = []
        self.drawing = False
        self.draw_start = None
        self.draw_rect = None
        self.status = None
        # 多边形标注
        self.poly_mode = False  # 多边形模式下按 p 切换
        self.poly_points = []   # 当前多边形的顶点
        self.poly_line = None   # 预览线
        self.poly_patches = []  # 已保存的多边形（用掩码存储）
        self.poly_mask_list = [[] for _ in range(100)]  # 每张图的多边形掩码列表

        # 颜色
        self.colors = plt.cm.tab10

    def run(self):
        print("\n🖱️  请在每张图上拖动左键画框标注目标（如珠光体）")
        print("   点击已标注的框可删除\n")
        self._start_gui()

    def _start_gui(self):
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        pass  # no buttons needed
        self.fig.canvas.manager.set_window_title("SAED Annotation Tool")
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._help_text = self.fig.text(0.5, 0.01,
            "左键拖动画框 | 点击框内删除 | i:多边形模式(连续点选) | Enter:闭合当前 | 右键:退出模式 | r:分割 | s:保存 | n/p:翻页 | q:退出",
            ha="center", fontsize=9, bbox=dict(boxstyle="round", facecolor="lightyellow"))

        self._show(self.idx)
        plt.show()

    def _show(self, idx):
        self.idx = idx
        path = self.paths[idx]
        self.img = np.array(Image.open(path).convert("RGB"))
        self._redraw_with_masks()

    def _update_status(self):
        n = len(self.boxes[self.idx])
        txt = f"图片 {self.idx+1}/{len(self.paths)} | 标注框: {n} | [n]下一张 [p]上一张 [s]保存 [q]退出"
        if self.status: self.status.remove()
        self.status = self.fig.text(0.5, 0.025, txt, ha="center", fontsize=10,
                                    bbox=dict(boxstyle="round", facecolor="lightyellow"))

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1: return
        x, y = event.xdata, event.ydata

        # 多边形模式：添加顶点
        if self.poly_mode:
            self.poly_points.append((x, y))
            self.ax.plot(x, y, 'ro', markersize=4, picker=False)[0]
            # 画连线
            if len(self.poly_points) > 1:
                lx, ly = zip(*self.poly_points)
                self.ax.plot(lx, ly, 'r-', linewidth=1.5, alpha=0.7)[0]
            self.fig.canvas.draw_idle()
            return

        # 检查是否点击了已有框（删除）
        for i, (bx, by, bw, bh) in enumerate(self.boxes[self.idx]):
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self.boxes[self.idx].pop(i)
                self._show(self.idx)
                return

        # 检查是否点击了已有分割结果（删除）
        masks_raw = self.masks[self.idx]
        if masks_raw is not None and len(masks_raw) > 0:
            h, w = self.img.shape[:2]
            for i in range(len(masks_raw) - 1, -1, -1):  # 倒序遍历
                mask = masks_raw[i]
                if isinstance(mask, np.ndarray) and mask.shape == (h, w):
                    if 0 <= int(y) < h and 0 <= int(x) < w and mask[int(y), int(x)]:
                        # 点击在掩码上 → 删除
                        if isinstance(masks_raw, list):
                            self.masks[self.idx].pop(i)
                        else:
                            self.masks[self.idx] = np.delete(masks_raw, i, axis=0)
                        print(f"删除分割实例 #{i+1}")
                        self._redraw_with_masks()
                        return

        # 开始画新框
        self.drawing = True
        self.draw_start = (x, y)
        self.draw_rect = Rectangle((x, y), 0, 0, fill=False, edgecolor="cyan",
                                    linewidth=2, linestyle="--")
        self.ax.add_patch(self.draw_rect)

    def _on_move(self, event):
        if not self.drawing or not self.draw_rect or event.xdata is None: return
        x0, y0 = self.draw_start
        x, y = event.xdata, event.ydata
        self.draw_rect.set_xy((min(x0, x), min(y0, y)))
        self.draw_rect.set_width(abs(x - x0))
        self.draw_rect.set_height(abs(y - y0))
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if not self.drawing or not self.draw_rect: return
        self.drawing = False
        x0, y0 = self.draw_start
        x, y = event.xdata, event.ydata
        w, h = abs(x - x0), abs(y - y0)
        if w > 5 and h > 5:
            self.boxes[self.idx].append([min(x0, x), min(y0, y), w, h])
        if self.draw_rect in self.ax.patches:
            self.draw_rect.remove()
        self.draw_rect = None
        self._show(self.idx)

    def _run_sam3(self):
        """用 SAM 3 分割框标注 + 直接使用多边形标注"""
        model, processor, device = self._get_model()
        h, w = self.img.shape[:2]
        img_pil = Image.fromarray(self.img)

        all_masks = []
        n_poly = 0

        # ---- 多边形：直接作为结果 ----
        for pmask in self.poly_mask_list[self.idx]:
            all_masks.append(pmask)
            n_poly += 1

        # ---- 框：逐个分割（每个框独立推理，避免 queries 竞争） ----
        for bx, by, bw, bh in self.boxes[self.idx]:
            # 裁剪图像到框区域（扩大 20% 边界）
            margin = 0.2
            x1 = max(0, bx - bw * margin)
            y1 = max(0, by - bh * margin)
            x2 = min(w, bx + bw + bw * margin)
            y2 = min(h, by + bh + bh * margin)
            crop = img_pil.crop((x1, y1, x2, y2))
            # 确保裁剪图不小于 224x224（ViT backbone 最小尺寸）
            if crop.size[0] < 224 or crop.size[1] < 224:
                crop = crop.resize((max(224, crop.size[0]), max(224, crop.size[1])))
            crop_w, crop_h = crop.size

            # 用 SAM 3 分割这个裁剪区域
            state = processor.set_image(crop)
            text_outputs = model.backbone.forward_text(["object"], device=device)
            state["backbone_out"].update(text_outputs)
            state["geometric_prompt"] = model._get_dummy_prompt()

            # 在裁剪图上添加框提示（框的位置就是裁剪后的目标位置）
            # 框占裁剪图的大部分区域，中心偏一点
            box_cx, box_cy = (bx - x1 + bw/2) / crop_w, (by - y1 + bh/2) / crop_h
            box_nw, box_nh = bw / crop_w * 0.8, bh / crop_h * 0.8  # 缩小一点避免贴边
            state["geometric_prompt"].append_boxes(
                torch.tensor([box_cx, box_cy, box_nw, box_nh], device=device, dtype=torch.float32).view(1, 1, 4),
                torch.tensor([True], device=device, dtype=torch.bool).view(1, 1),
            )

            output = model.forward_grounding(
                backbone_out=state["backbone_out"],
                find_input=processor.find_stage,
                geometric_prompt=state["geometric_prompt"],
                find_target=None,
            )
            out_masks = output["pred_masks"]
            out_scores = output["pred_logits"].sigmoid()
            presence = output["presence_logit_dec"].sigmoid().unsqueeze(1)
            keep = (out_scores * presence).squeeze(-1) > 0.05  # 更低阈值
            for i in range(len(out_masks[keep])):
                m = out_masks[keep][i].detach().cpu().numpy().squeeze()
                if m.shape != (int(crop_h), int(crop_w)):
                    from skimage.transform import resize
                    m = resize(m.astype(float), (int(crop_h), int(crop_w)), preserve_range=True) > 0.5
                # 将裁剪的掩码映射回原图坐标
                full_mask = np.zeros((h, w), dtype=bool)
                x1i, y1i = int(x1), int(y1)
                x2i, y2i = x1i + m.shape[1], y1i + m.shape[0]
                x2i, y2i = min(x2i, w), min(y2i, h)
                mh, mw = y2i - y1i, x2i - x1i
                full_mask[y1i:y2i, x1i:x2i] = m[:mh, :mw]
                all_masks.append(full_mask)
            # 清理显存
            del state, output, out_masks, out_scores
            if device == "cuda":
                torch.cuda.empty_cache()

        # ---- 合并结果 ----
        if all_masks:
            self.masks[self.idx] = all_masks  # list of masks (may have different shapes)
            print(f"✅ {n_poly} 个多边形 + {len(self.boxes[self.idx])} 个框分割结果")
        else:
            print("⚠️  无结果")
        self._redraw_with_masks()  # full redraw

    def _redraw_with_masks(self):
        """重绘图像 + 框 + 遮罩"""
        self.ax.clear()
        self.rect_patches.clear()
        self.ax.imshow(self.img)
        self.ax.set_title(f"[{self.idx+1}/{len(self.paths)}] {os.path.basename(self.paths[self.idx])}", fontsize=13)
        self.ax.axis("on")

        # 画遮罩
        masks_raw = self.masks[self.idx]
        if masks_raw is not None and len(masks_raw) > 0:
            h, w = self.img.shape[:2]
            # 统一为列表格式
            if isinstance(masks_raw, np.ndarray):
                if masks_raw.ndim == 4: masks_raw = masks_raw[:, 0]
                if masks_raw.ndim == 2: masks_raw = [masks_raw]
                else: masks_raw = list(masks_raw)
            masks_np = masks_raw
            overlay = np.zeros((h, w, 4), dtype=np.float32)
            for i in range(len(masks_np)):
                mask = masks_np[i]
                if mask.shape != (h, w):
                    from skimage.transform import resize
                    mask = resize(mask.astype(float), (h, w), preserve_range=True) > 0.5
                color = self.colors(i % 10)[:3]
                for c in range(3):
                    overlay[mask, c] = overlay[mask, c] * 0.7 + color[c] * 0.3
                overlay[mask, 3] = np.maximum(overlay[mask, 3], 0.5)
            overlay[:, :, 3] = np.clip(overlay[:, :, 3], 0, 1)
            self.ax.imshow(overlay)

        # 画框
        for box in self.boxes[self.idx]:
            x, y, wb, hb = box
            color = self.colors(len(self.rect_patches) % 10)
            r = Rectangle((x, y), wb, hb, fill=False, edgecolor=color, linewidth=2, alpha=0.8)
            self.ax.add_patch(r)
            self.rect_patches.append(r)

        # 画已保存的多边形轮廓
        for pmask in self.poly_mask_list[self.idx]:
            from skimage.measure import find_contours
            for c in find_contours(pmask.astype(float), 0.5):
                self.ax.plot(c[:, 1], c[:, 0], 'r-', linewidth=2, alpha=0.8)

        self._update_status()
        self.fig.canvas.draw_idle()

    def _get_model(self):
        """获取/构建 SAM 3 模型（单例）"""
        if not hasattr(self, "_model") or self._model is None:
            ckpt = os.path.join(SAM3_ROOT, "checkpoints", "sam3.pt")
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"\n🔄 构建 SAM 3 模型（首次需 ~6 秒）...")
            self._model = build_sam3_image_model(
                checkpoint_path=ckpt, device=dev,
                eval_mode=True, enable_segmentation=True, enable_inst_interactivity=False,
            )
            self._processor = Sam3Processor(self._model, resolution=1008, device=dev,
                                            confidence_threshold=0.3)
            self._device = dev
            print("✅ 模型就绪\n")
        return self._model, self._processor, self._device

    def _save(self):
        """保存当前标注"""
        idx = self.idx
        name = os.path.basename(self.paths[idx])
        h, w = self.img.shape[:2]
        total_area = h * w

        # 计算框内面积（粗略估算）
        box_area = sum(bw * bh for _, _, bw, bh in self.boxes[idx])

        data = {
            "image": name,
            "image_size": (w, h),
            "n_boxes": len(self.boxes[idx]),
            "area_total_pixels": total_area,
            "area_annotated_pixels": box_area,
            "area_percentage": round(box_area / total_area * 100, 4) if total_area else 0,
            "boxes": [{"x": b[0], "y": b[1], "w": b[2], "h": b[3]} for b in self.boxes[idx]],
        }

        save_path = os.path.join(self.output_dir, f"{os.path.splitext(name)[0]}.json")
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ 已保存: {save_path} (标注面积占比 {data['area_percentage']:.2f}%)")

    def _on_key(self, event):
        if event.key == "n":
            self._save()
            if self.idx < len(self.paths) - 1:
                self.idx += 1
                self._show(self.idx)
        elif event.key == "p":
            if self.idx > 0:
                self.idx -= 1
                self._show(self.idx)
        elif event.key == "i":
            self.poly_mode = not self.poly_mode
            if self.poly_mode:
                self.poly_points = []
                print("多边形模式: 点击目标边缘添加顶点，Enter 闭合")
            else:
                print("退出多边形模式")
        elif event.key == "enter":
            if self.poly_mode and len(self.poly_points) > 2:
                # 闭合多边形，转换为掩码
                from skimage.draw import polygon2mask
                h, w = self.img.shape[:2]
                pts = np.array(self.poly_points)
                mask = polygon2mask((h, w), pts[:, ::-1])  # (y, x)
                # 保存掩码
                self.poly_mask_list[self.idx].append(mask)
                n = len(self.poly_mask_list[self.idx])
                print(f"多边形 #{n} 已保存 ({int(mask.sum())} 像素)")
                self.poly_points = []  # 清空，继续画下一个
                self._redraw_with_masks()
            else:
                print("多边形至少需要 3 个顶点")
        elif event.key == "r":
            self._run_sam3()
        elif event.key == "s":
            self._save()
        elif event.key == "q":
            self._quit()

    def _quit(self):
        self._save()
        plt.close("all")
        print("👋 退出")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join(SAM3_ROOT, "data", "3"))
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    Annotator(args.data_dir, args.output_dir).run()
