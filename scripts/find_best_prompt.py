#!/usr/bin/env python3
"""
从已有标注中学习视觉特征，找到最佳自然语言描述
==============================================
流程:
  1. 加载你手动标注的框
  2. 显示每个框的截图，让你视觉确认目标长什么样
  3. 尝试多种自然语言描述，看哪个效果最好
  4. 用最佳描述批量处理所有图片

用法:
  python scripts/find_best_prompt.py
  python scripts/find_best_prompt.py --show_crops  # 只看标注区域
  python scripts/find_best_prompt.py --test_prompts  # 测试提示词
"""

import sys, os, json, glob, argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SAM3_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SAM3_ROOT)

DEFAULT_ANN_DIR = os.path.join(SAM3_ROOT, "data", "3", "annotations")
DEFAULT_IMG_DIR = os.path.join(SAM3_ROOT, "data", "3")


def show_annotations(ann_dir=DEFAULT_ANN_DIR, img_dir=DEFAULT_IMG_DIR):
    """显示所有标注框的截图"""
    ann_files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
    if not ann_files:
        print(f"❌ {ann_dir} 中没有标注文件")
        return

    print(f"找到 {len(ann_files)} 个标注文件\n")

    for ann_file in ann_files:
        with open(ann_file) as f:
            ann = json.load(f)

        img_path = os.path.join(img_dir, ann["image"])
        if not os.path.exists(img_path):
            print(f"⚠️  找不到图片: {ann['image']}")
            continue

        img = np.array(Image.open(img_path).convert("RGB"))
        h, w = img.shape[:2]

        print(f"\n{'='*50}")
        print(f"📷 {ann['image']}")
        print(f"   框数: {ann['n_boxes']}")
        print(f"   标注面积占比: {ann.get('area_percentage', 'N/A')}%")

        # 显示原图 + 框
        fig, axes = plt.subplots(1, min(3, ann["n_boxes"]) + 1, figsize=(16, 5))
        if ann["n_boxes"] == 0:
            continue

        # 原图
        axes[0].imshow(img)
        axes[0].set_title(f"原图 ({w}x{h})", fontsize=10)
        axes[0].axis("off")
        for b in ann["boxes"]:
            axes[0].add_patch(Rectangle((b["x"], b["y"]), b["w"], b["h"],
                                        fill=False, edgecolor="lime", linewidth=2))

        # 每个框的截图
        for i, b in enumerate(ann["boxes"][:3]):
            x, y, bw, bh = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
            crop = img[y:y + bh, x:x + bw]
            axes[i + 1].imshow(crop)
            axes[i + 1].set_title(f"目标 {i+1}: {bw}x{bh}", fontsize=10)
            axes[i + 1].axis("off")

        plt.suptitle(f"观察目标特征，用于选择自然语言描述", fontsize=13)
        plt.tight_layout()
        plt.show()


def test_prompts(ann_dir=DEFAULT_ANN_DIR, img_dir=DEFAULT_IMG_DIR):
    """用不同自然语言描述测试分割效果"""
    from sam3.sam3_detect_saed_spots import detect_saed_spots

    # 候选提示词（根据你的目标类型调整）
    prompts = [
        "dark etched region",
        "irregular dark cluster",
        "pearlite colony at grain boundary",
        "dark lamellar structure",
        "dark phase at boundary",
        "dark irregular object",
    ]

    # 用第一张标注图来测试
    ann_files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
    if not ann_files:
        print(f"❌ {ann_dir} 中没有标注文件")
        return

    with open(ann_files[0]) as f:
        ann = json.load(f)
    img_path = os.path.join(img_dir, ann["image"])
    if not os.path.exists(img_path):
        print(f"❌ 找不到图片: {img_path}")
        return

    img = np.array(Image.open(img_path).convert("RGB"))
    h, w = img.shape[:2]

    # 计算标注框的总面积作为参考
    ref_area = sum(b["w"] * b["h"] for b in ann["boxes"])
    ref_pct = ref_area / (w * h) * 100

    print(f"\n测试图片: {ann['image']}")
    print(f"手动标注面积占比: {ref_pct:.2f}%\n")

    # 显示手动标注
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.set_title("手动标注（参考）", fontsize=12)
    for b in ann["boxes"]:
        ax.add_patch(Rectangle((b["x"], b["y"]), b["w"], b["h"],
                               fill=False, edgecolor="green", linewidth=2))
    ax.axis("off")
    plt.show()

    print(f"{'提示词':<45} | {'目标数':>6} | {'面积占比':>8}")
    print("-" * 65)

    results = []
    for prompt in prompts:
        try:
            result = detect_saed_spots(img_path, prompt=prompt, verbose=False)
            n = result["n_spots"]
            # 近似面积占比（用所有检测框的面积估算）
            area_pct = 0
            if n > 0 and "boxes" in result:
                pass  # 精确面积需用 masks
            print(f"{prompt:<45} | {n:>6} |  --")
            results.append((prompt, n, result))
        except Exception as e:
            print(f"{prompt:<45} |  错误 | {e}")

    print(f"\n参考: 手动标注 {len(ann['boxes'])} 个框, 面积占比 ~{ref_pct:.1f}%")

    # 选出效果最好的提示词
    if results:
        print(f"\n建议对最佳结果使用 'r' 键手动微调后保存。")
        print(f"然后用选定的提示词批量处理：")
        print(f"  detect_saed_spots('new_image.jpg', prompt='{results[0][0]}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_dir", default=DEFAULT_ANN_DIR)
    parser.add_argument("--img_dir", default=DEFAULT_IMG_DIR)
    parser.add_argument("--show_crops", action="store_true", help="只看标注区域")
    parser.add_argument("--test_prompts", action="store_true", help="测试提示词")
    args = parser.parse_args()

    if args.show_crops:
        show_annotations(args.ann_dir, args.img_dir)
    elif args.test_prompts:
        test_prompts(args.ann_dir, args.img_dir)
    else:
        # 默认：先看标注，再测提示词
        show_annotations(args.ann_dir, args.img_dir)
        input("\n按 Enter 继续测试提示词...")
        test_prompts(args.ann_dir, args.img_dir)
