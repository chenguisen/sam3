#!/usr/bin/env python3
"""
pearlite_lath_spacing.py — SEM 板条组织片层间距自动测量（FFT 方法）

利用 FFT 幅度谱中对称衍射点对检测板条周期结构：
  1. 加载图像 → 计算 2D FFT
  2. 检测对称衍射点对 → 计算层间距 d = 1 / sqrt((kx/W)^2 + (ky/H)^2)
  3. 频域滤波 → 逆 FFT → 空间定位各衍射对的 mask
  4. 保存结果

Usage:
  python pearlite_lath_spacing.py <image_path> [--crop-bottom 80] [--save-dir results]

Dependencies:
  numpy, PIL, matplotlib, scikit-image, scipy
"""

import sys, os, json, time, argparse
import numpy as np
from PIL import Image
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter, binary_closing
from scipy import ndimage

matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
#  参数
# ============================================================

# --- 图像 ---
CROP_BOTTOM = 80

# --- FFT 衍射点检测 ---
FFT_SIGMA = 2.0                # FFT 平滑 sigma
MIN_PEAK_DIST_FFT = 15         # 峰值间最小距离
PEAK_REL_TH = 0.75             # 相对阈值
TOP_N_PEAKS = 15               # 保留最强峰数量
SYM_TOLERANCE = 3.0            # 对称匹配容差
EXCLUDE_CENTER_RADIUS = 5      # 排除中心区域
MAX_DETECTIONS = 5             # 最多显示对数

# --- 空间定位 ---
FILTER_SIGMA = 3.0             # 频域高斯滤波器 sigma
MASK_THRESHOLD = 0.3           # 空间 mask 阈值
OVERLAY_ALPHA = 0.45           # mask 透明度

# --- 保存 ---
SAVE_DPI = 300


# ============================================================
#  函数
# ============================================================

def load_image(image_path, crop_bottom=0):
    """加载图像，返回 (img_rgb, img_gray, h, w)"""
    img_pil = Image.open(image_path).convert('RGB')
    if crop_bottom > 0:
        img_pil = img_pil.crop((0, 0, img_pil.width, img_pil.height - crop_bottom))
    img = np.array(img_pil)
    img_gray = np.mean(img, axis=2).astype(float)
    h, w, _ = img.shape
    print(f'Image: {Path(image_path).name}  Size: {w}x{h}  '
          f'Gray mean={img_gray.mean():.1f}  range=[{img_gray.min():.0f}, {img_gray.max():.0f}]')
    return img, img_gray, h, w


def compute_fft(img_gray):
    """计算 2D FFT，返回 fshift, magnitude_spectrum"""
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    return fshift, magnitude_spectrum


def find_symmetric_pairs(mag_smoothed, h, w,
                         min_distance=15, threshold_rel=0.75,
                         top_n_peaks=15, sym_tolerance=3.0,
                         exclude_center_radius=5):
    """
    在平滑后的 FFT 幅度谱中检测对称峰值对。

    Returns:
      pairs: list of (y1, x1, y2, x2, r, angle, kx, ky, d)
      peaks: detected peak coordinates
    """
    h_f, w_f = mag_smoothed.shape
    cy_f, cx_f = h_f // 2, w_f // 2

    peaks = peak_local_max(mag_smoothed, min_distance=min_distance,
                           threshold_rel=threshold_rel, exclude_border=False)

    # 排除中心附近
    dist_center = np.sqrt((peaks[:, 0] - cy_f)**2 + (peaks[:, 1] - cx_f)**2)
    peaks = peaks[dist_center > exclude_center_radius]
    print(f"  FFT peaks after center exclusion: {len(peaks)}")

    # 按强度排序，保留最强
    if len(peaks) == 0:
        return [], peaks

    if len(peaks) > top_n_peaks:
        peak_values = mag_smoothed[peaks[:, 0], peaks[:, 1]]
        top_idx = np.argsort(peak_values)[::-1][:top_n_peaks]
        peaks = peaks[top_idx]
    print(f"  Top {len(peaks)} peaks kept for matching")

    # 寻找对称对
    used = set()
    pairs = []
    for i, (y1, x1) in enumerate(peaks):
        if i in used:
            continue
        y2_sym = 2 * cy_f - y1
        x2_sym = 2 * cx_f - x1
        dists = np.sqrt((peaks[:, 0] - y2_sym)**2 + (peaks[:, 1] - x2_sym)**2)
        dists[list(used)] = 9999
        dists[i] = 9999
        nearest = np.argmin(dists)
        if dists[nearest] <= sym_tolerance:
            j = nearest
            used.add(i)
            used.add(j)
            kx = x1 - cx_f
            ky = y1 - cy_f
            r = np.sqrt(kx**2 + ky**2)
            angle = np.rad2deg(np.arctan2(ky, kx))
            d = 1.0 / np.sqrt((kx / w)**2 + (ky / h)**2)
            pairs.append((y1, x1, 2*cy_f-y1, 2*cx_f-x1, r, angle, kx, ky, d))

    pairs.sort(key=lambda p: p[4])
    print(f"  Found {len(pairs)} symmetric pairs")
    return pairs, peaks


def localize_pairs(pairs, fshift, h, w, filter_sigma=3.0, mask_threshold=0.3):
    """
    对每对衍射点做频域滤波 → 逆 FFT → 空间 mask。
    Returns: list of (mask, kx, ky, d, angle)
    """
    h_f, w_f = fshift.shape
    cy_f, cx_f = h_f // 2, w_f // 2
    Y, X = np.ogrid[:h_f, :w_f]
    ky_grid = Y - cy_f
    kx_grid = X - cx_f

    results = []
    for y1, x1, y2, x2, r, ang, kx_val, ky_val, d_val in pairs:
        d1 = np.sqrt((kx_grid - kx_val)**2 + (ky_grid - ky_val)**2)
        d2 = np.sqrt((kx_grid + kx_val)**2 + (ky_grid + ky_val)**2)
        filt = np.exp(-0.5 * (d1 / filter_sigma)**2) + np.exp(-0.5 * (d2 / filter_sigma)**2)
        filt = filt / filt.max()

        filtered = fshift * filt
        spatial_map = np.fft.ifft2(np.fft.ifftshift(filtered))
        spatial_mag = np.abs(spatial_map)

        smin, smax = spatial_mag.min(), spatial_mag.max()
        spatial_norm = (spatial_mag - smin) / (smax - smin + 1e-10)
        mask = spatial_norm > mask_threshold
        mask = binary_closing(mask, iterations=2)

        results.append((mask, kx_val, ky_val, d_val, ang))
    return results


def save_results(save_dir, name_stem, img, pairs, local_results,
                 fshift, magnitude_spectrum, radial_profile,
                 no_figures=False):
    """保存所有结果"""
    os.makedirs(save_dir, exist_ok=True)
    h, w = img.shape[:2]
    h_f, w_f = magnitude_spectrum.shape
    cy_f, cx_f = h_f // 2, w_f // 2

    # ---- JSON summary ----
    summary = {
        'image': name_stem,
        'image_size_px': [w, h],
        'fft_params': {
            'sigma': FFT_SIGMA,
            'min_distance': MIN_PEAK_DIST_FFT,
            'threshold_rel': PEAK_REL_TH,
            'top_n_peaks': TOP_N_PEAKS,
            'sym_tolerance': SYM_TOLERANCE,
        },
        'n_pairs': len(pairs),
        'pairs': [],
    }
    for idx, (_, _, _, _, r, ang, kx, ky, d) in enumerate(pairs):
        lath_dir = (ang - 90) % 180
        summary['pairs'].append({
            'id': idx + 1,
            'kx_px': int(kx),
            'ky_px': int(ky),
            'radius': round(r, 1),
            'angle_deg': round(ang, 1),
            'lath_direction_deg': round(lath_dir, 1),
            'spacing_px': round(d, 1),
        })
    # 空间定位面积
    if local_results:
        for idx, (mask, _, _, d_val, ang) in enumerate(local_results):
            if idx < len(summary['pairs']):
                summary['pairs'][idx]['area_px'] = int(mask.sum())
                summary['pairs'][idx]['area_pct'] = round(mask.sum() / (h*w) * 100, 2)

    with open(os.path.join(save_dir, f'{name_stem}_fft_spacing.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    if no_figures:
        print(f'  JSON: {name_stem}_fft_spacing.json')
        return

    # ---- 图1: 原始图像 + FFT 幅度谱 ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(img if img.ndim == 3 else img, cmap='gray')
    axes[0].set_title(f'Original ({w}x{h})', fontsize=12)
    axes[0].axis('off')
    im = axes[1].imshow(magnitude_spectrum, cmap='inferno', interpolation='bilinear')
    axes[1].set_title('FFT Magnitude Spectrum', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, label='log magnitude')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{name_stem}_fft_spectrum.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图2: 径向平均功率谱 ----
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(radial_profile, 'b-', linewidth=1.5)
    ax2.set_title('Radial Average of Power Spectrum', fontsize=12)
    ax2.set_xlabel('Distance from center (frequency)')
    ax2.set_ylabel('Mean log magnitude')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(os.path.join(save_dir, f'{name_stem}_fft_radial.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)

    if len(pairs) == 0:
        return

    # ---- 图3: FFT + 峰值标记（2x2 布局） ----
    mag_smoothed = gaussian_filter(magnitude_spectrum, sigma=FFT_SIGMA)
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(pairs), 10)))
    _, peaks = find_symmetric_pairs(mag_smoothed, h, w,
                                     min_distance=MIN_PEAK_DIST_FFT,
                                     threshold_rel=PEAK_REL_TH,
                                     top_n_peaks=TOP_N_PEAKS,
                                     sym_tolerance=SYM_TOLERANCE,
                                     exclude_center_radius=EXCLUDE_CENTER_RADIUS)

    fig = plt.figure(figsize=(18, 14))
    # (0,0) 原始 FFT
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(magnitude_spectrum, cmap='inferno', interpolation='bilinear')
    ax.set_title('Original FFT spectrum', fontsize=20)
    ax.axis('off')
    iax = ax.inset_axes([0.68, 0.68, 0.30, 0.30])
    iax.imshow(mag_smoothed, cmap='inferno', interpolation='bilinear')
    iax.set_title('Smoothed', fontsize=14, color='white')
    iax.axis('off')

    # (0,1) 平滑 FFT + 峰值
    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(mag_smoothed, cmap='inferno', interpolation='bilinear')
    if len(peaks) > 0:
        ax.scatter(peaks[:, 1], peaks[:, 0], c='cyan', s=16, alpha=0.6, label=f'{len(peaks)} peaks')
    for idx, (y1, x1, y2, x2, r, ang, kx, ky, d) in enumerate(pairs[:10]):
        c = colors[idx]
        ax.scatter([x1, x2], [y1, y2], c=[c], s=160, edgecolors='white', linewidths=2,
                   zorder=5, label=f'Pair {idx+1}: k=({kx:.0f},{ky:.0f}) d={d:.0f}px' if idx == 0 else '')
        ax.plot([x1, x2], [y1, y2], c=c, linewidth=2.5, linestyle='--', alpha=0.7)
    ax.scatter(cx_f, cy_f, c='white', s=250, marker='+', linewidths=3, zorder=10, label='Center')
    ax.set_title(f'Smoothed FFT (sigma={FFT_SIGMA})\nwith symmetric pairs', fontsize=20)
    ax.axis('off')
    ax.legend(fontsize=12, loc='upper right')

    # (1,0) 极坐标
    if len(pairs) > 0:
        radii = np.array([p[4] for p in pairs])
        angles_deg = np.array([p[5] for p in pairs])
        angles_rad = np.deg2rad(angles_deg)
        ax = fig.add_subplot(2, 2, 3, projection='polar')
        sc = ax.scatter(angles_rad, radii, c=radii, cmap='plasma',
                        s=120, alpha=0.9, edgecolors='white', linewidths=1, zorder=5)
        ax.set_title('Diffraction spots (polar coords)', fontsize=20, pad=20)
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(-1)
        plt.colorbar(sc, ax=ax, fraction=0.08, label='Radius')
        for i in range(min(len(pairs), 8)):
            a = angles_rad[i]
            r = radii[i]
            lath_dir = (angles_deg[i] - 90) % 180
            ax.annotate(f'{lath_dir:.0f}deg', xy=(a, r),
                        fontsize=14, ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    else:
        ax = fig.add_subplot(2, 2, 3)
        ax.text(0.5, 0.5, 'No pairs', ha='center', va='center', fontsize=20, transform=ax.transAxes)

    # (1,1) 原始图像 + 方向线
    ax = fig.add_subplot(2, 2, 4)
    ax.imshow(img if img.ndim == 3 else img, cmap='gray')
    for idx, (y1, x1, y2, x2, r, ang, kx, ky, d) in enumerate(pairs[:10]):
        c = colors[idx]
        lath_dir_deg = (ang - 90) % 180
        lath_rad = np.deg2rad(lath_dir_deg)
        length = min(h, w) * 0.35
        cx_img, cy_img = w // 2, h // 2
        dx_line = length * np.cos(lath_rad)
        dy_line = -length * np.sin(lath_rad)
        ax.plot([cx_img - dx_line, cx_img + dx_line],
                [cy_img - dy_line, cy_img + dy_line],
                c=c, linewidth=3, linestyle='-', alpha=0.8,
                label=f'Pair {idx+1}: {lath_dir_deg:.0f}deg / d={d:.0f}px' if idx == 0 else '')
    ax.legend(fontsize=14, loc='upper right')
    ax.set_title('Detected lath orientations from FFT', fontsize=20)
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{name_stem}_fft_pairs.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 图4: 空间定位 mask 叠加 ----
    if local_results:
        fig2, ax2 = plt.subplots(figsize=(14, 10))
        ax2.imshow(img, interpolation='bilinear')
        for idx, (mask_i, _, _, d_val, ang) in enumerate(local_results):
            c = colors[idx]
            overlay = np.zeros((h, w, 4))
            overlay[mask_i] = [*c[:3], OVERLAY_ALPHA]
            ax2.imshow(overlay)
            ys, xs = np.where(mask_i)
            if len(ys) > 0:
                lath_dir = (ang - 90) % 180
                ax2.text(xs.mean(), ys.mean(),
                        f'Pair {idx+1}\n{d_val:.0f}px\n{lath_dir:.0f}deg',
                        color='white', fontsize=16, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=c, alpha=0.7))
        ax2.set_title('Spatial localization of diffraction pairs', fontsize=16)
        ax2.axis('off')
        plt.tight_layout()
        fig2.savefig(os.path.join(save_dir, f'{name_stem}_fft_localization.png'),
                     dpi=150, bbox_inches='tight')
        plt.close(fig2)

    print(f'  Saved figures to {save_dir}/')


# ============================================================
#  主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='SEM 板条组织片层间距自动测量（FFT 方法）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('image_path', type=str, help='Path to input image (TIFF/PNG/JPG)')
    parser.add_argument('--crop-bottom', type=int, default=CROP_BOTTOM,
                        help=f'Pixels to crop from bottom (default: {CROP_BOTTOM})')
    parser.add_argument('--save-dir', type=str, default='results',
                        help='Output directory (default: results)')
    parser.add_argument('--no-figures', action='store_true',
                        help='Only save JSON, skip generating figures')
    args = parser.parse_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    name_stem = Path(args.image_path).stem

    t_start = time.time()

    # ==================== 1. 加载图像 ====================
    print('=' * 55)
    print('  Loading image...')
    print('=' * 55)
    img, img_gray, h, w = load_image(args.image_path, args.crop_bottom)

    # ==================== 2. FFT 变换 ====================
    print('\n' + '=' * 55)
    print('  FFT Analysis')
    print('=' * 55)
    fshift, magnitude_spectrum = compute_fft(img_gray)
    h_f, w_f = magnitude_spectrum.shape
    cy_f, cx_f = h_f // 2, w_f // 2

    # 径向平均
    Y_grid, X_grid = np.ogrid[:h_f, :w_f]
    dist_from_center = np.sqrt((Y_grid - cy_f)**2 + (X_grid - cx_f)**2)
    max_radius = min(cy_f, cx_f)
    radial_profile = np.zeros(max_radius)
    for r in range(1, max_radius):
        mask = (dist_from_center >= r - 0.5) & (dist_from_center < r + 0.5)
        radial_profile[r] = magnitude_spectrum[mask].mean()

    # ==================== 3. 衍射点检测 ====================
    print('\n' + '-' * 55)
    print('  Diffraction Spot Detection')
    print('-' * 55)
    mag_smoothed = gaussian_filter(magnitude_spectrum, sigma=FFT_SIGMA)
    pairs, peaks = find_symmetric_pairs(
        mag_smoothed, h, w,
        min_distance=MIN_PEAK_DIST_FFT,
        threshold_rel=PEAK_REL_TH,
        top_n_peaks=TOP_N_PEAKS,
        sym_tolerance=SYM_TOLERANCE,
        exclude_center_radius=EXCLUDE_CENTER_RADIUS,
    )

    if len(pairs) > 0:
        print(f'\n  Diffraction spot summary:')
        print(f'  {"#":>3s}  {"kx":>6s}  {"ky":>6s}  {"Radius":>7s}  '
              f'{"Angle(deg)":>10s}  {"Lath dir":>9s}  {"Spacing(px)":>10s}')
        print('  ' + '-' * 65)
        for idx, (_, _, _, _, r, ang, kx, ky, d) in enumerate(pairs):
            lath_dir = (ang - 90) % 180
            print(f'  {idx+1:3d}  {kx:6.0f}  {ky:6.0f}  {r:7.1f}  '
                  f'{ang:10.1f}  {lath_dir:9.1f}  {d:10.1f}')

        # 空间定位
        print('\n  Spatial localization...')
        local_results = localize_pairs(pairs, fshift, h, w,
                                       filter_sigma=FILTER_SIGMA,
                                       mask_threshold=MASK_THRESHOLD)

        print(f'\n  Localization summary:')
        print(f'  {"Pair":>5s}  {"Area(px)":>9s}  {"Area(%)":>8s}  '
              f'{"Spacing":>8s}  {"Lath dir":>9s}')
        print('  ' + '-' * 45)
        for idx, (mask, _, _, d_val, ang) in enumerate(local_results):
            lath_dir = (ang - 90) % 180
            print(f'  {idx+1:5d}  {mask.sum():9d}  {mask.sum()/(h*w)*100:7.1f}%  '
                  f'{d_val:8.0f}px  {lath_dir:8.0f}deg')
    else:
        print('  No symmetric pairs found.')
        local_results = []

    # ==================== 4. 保存 ====================
    print('\n' + '=' * 55)
    print('  Saving Results')
    print('=' * 55)
    save_results(save_dir, name_stem, img, pairs, local_results,
                 fshift, magnitude_spectrum, radial_profile,
                 no_figures=args.no_figures)

    elapsed = time.time() - t_start
    print(f'\nTotal time: {elapsed:.1f}s')
    print(f'Output: {save_dir}/')
    print('Done!')


if __name__ == '__main__':
    main()
