#!/usr/bin/env python3
"""
DD5 Alloy Segmentation GUI — Qt6

阈值策略 + 梯度 Watershed 的交互式调参工具。
后台线程运行，界面不卡顿，带日志面板。
"""

import sys, os, json, time
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
from skimage import filters
from skimage.measure import label, regionprops
from skimage.segmentation import watershed, find_boundaries
from skimage import morphology
from scipy import ndimage as ndi
import colorsys

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QPixmap, QImage, QAction, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QComboBox, QSlider,
    QPushButton, QFileDialog, QGroupBox, QStatusBar,
    QMessageBox, QSizePolicy, QPlainTextEdit, QDockWidget, QSplitter,
    QFrame,
)

# ====================================================================
#  Processing logic (runs in worker thread)
# ====================================================================

THRESHOLD_METHODS = [
    'otsu', 'yen', 'triangle', 'li',
    'isodata', 'mean', 'minimum', 'multiotsu',
]


def segment_threshold(img_gray, method='otsu', sigma=0.8, th_coef=1.0,
                      open_iter=1, min_area=30, expand=0.10,
                      use_watershed=False, dist_sigma=3.0):
    """Unified segmentation pipeline."""
    h, w = img_gray.shape

    if sigma > 0:
        smooth = filters.gaussian(img_gray, sigma=sigma)
    else:
        smooth = img_gray.copy()

    if method == 'otsu':
        th_base = filters.threshold_otsu(smooth)
    elif method == 'yen':
        th_base = filters.threshold_yen(smooth)
    elif method == 'triangle':
        th_base = filters.threshold_triangle(smooth)
    elif method == 'li':
        th_base = filters.threshold_li(smooth)
    elif method == 'isodata':
        th_base = filters.threshold_isodata(smooth)
    elif method == 'mean':
        th_base = filters.threshold_mean(smooth)
    elif method == 'minimum':
        th_base = filters.threshold_minimum(smooth)
    elif method == 'multiotsu':
        th_base = filters.threshold_multiotsu(smooth, classes=2)[0]
    else:
        raise ValueError(f'Unknown method: {method}')

    th_used = th_base * th_coef
    mask_bright = smooth > th_used

    if open_iter > 0:
        mask_bright = ndi.binary_opening(mask_bright, iterations=open_iter)

    if use_watershed:
        gradient = filters.sobel(smooth)
        dist = ndi.distance_transform_edt(mask_bright)
        dist_smooth = filters.gaussian(dist, sigma=dist_sigma)
        coords = morphology.local_maxima(dist_smooth)
        marker_mask = np.zeros_like(dist, dtype=int)
        marker_mask[coords] = 1
        markers, _ = ndi.label(marker_mask)
        labels = watershed(gradient, markers, mask=mask_bright)
        masks = []
        for lbl in range(1, labels.max() + 1):
            m = labels == lbl
            if m.sum() < min_area:
                continue
            m = ndi.binary_fill_holes(m)
            masks.append(m)
    else:
        labeled = label(mask_bright)
        props = regionprops(labeled)
        masks = []
        for p in props:
            if p.area < min_area:
                continue
            m = labeled == p.label
            m = ndi.binary_fill_holes(m)
            masks.append(m)

    stats = {
        'method': method,
        'sigma': sigma,
        'th_coef': th_coef,
        'open_iter': open_iter,
        'min_area': min_area,
        'th_used': float(th_used),
        'n_blocks': len(masks),
        'use_watershed': use_watershed,
        'dist_sigma': dist_sigma if use_watershed else None,
    }
    return masks, mask_bright, float(th_used), stats


def make_colored_overlay(img, masks):
    """Return (overlay_rgba, label_rgb_with_boundaries)."""
    h, w = img.shape[:2]
    n = len(masks)
    if n == 0:
        return np.zeros((h, w, 4), dtype=np.float32), img.copy()

    colors = np.array([
        colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, 0.9, 1.0)
        for i in range(n)
    ])

    overlay = np.zeros((h, w, 4), dtype=np.float32)
    for i in range(n):
        m = masks[i]
        for c in range(3):
            overlay[m, c] = overlay[m, c] * 0.7 + colors[i][c] * 0.3
        overlay[m, 3] = np.maximum(overlay[m, 3], 0.5)
    overlay[:, :, 3] = np.clip(overlay[:, :, 3], 0, 1)

    label_rgb = np.array(img, dtype=np.uint8).copy()
    for i in range(n):
        c255 = (colors[i][:3] * 255).astype(np.uint8)
        m = masks[i]
        for ch in range(3):
            label_rgb[m, ch] = np.clip(
                label_rgb[m, ch].astype(float) * 0.2 + c255[ch] * 0.8,
                0, 255
            ).astype(np.uint8)
    for i in range(n):
        b = find_boundaries(masks[i], mode='outer')
        label_rgb[b] = [255, 255, 255]

    return overlay, label_rgb


def compute_grain_stats(masks, img_shape):
    """Compute per-grain statistics: area, diameter, major/minor axis."""
    grains = []
    for i, m in enumerate(masks):
        ys, xs = np.where(m)
        area_px = int(m.sum())
        diam = np.sqrt(4 * area_px / np.pi)

        # Major & minor axis from regionprops
        labeled = label(m)
        props = regionprops(labeled)
        if props:
            major = props[0].major_axis_length
            minor = props[0].minor_axis_length
        else:
            major = minor = 0.0

        grains.append({
            'index': i + 1,
            'cx': float(np.mean(xs)),
            'cy': float(np.mean(ys)),
            'area': area_px,
            'area_pct': round(area_px / (img_shape[0] * img_shape[1]) * 100, 2),
            'diameter_px': round(diam, 1),
            'major_axis_px': round(float(major), 1),
            'minor_axis_px': round(float(minor), 1),
        })
    return grains


# ====================================================================
#  Worker thread
# ====================================================================

class WorkerSignals(QObject):
    finished = pyqtSignal(object)  # dict with results
    error    = pyqtSignal(str)
    log      = pyqtSignal(str)


class SegmentationWorker(QThread):
    def __init__(self, img_gray, img_rgb, params):
        super().__init__()
        self._img_gray = img_gray
        self._img_rgb = img_rgb
        self._params = params
        self.signals = WorkerSignals()

    def run(self):
        try:
            t0 = time.time()
            p = self._params
            self.signals.log.emit(
                f'[{datetime.now():%H:%M:%S}] ▶ Start: {p["method"]}'
                + (' (Watershed)' if p['use_watershed'] else '')
            )
            param_str = (f'sigma={p["sigma"]:.2f} th_coef={p["th_coef"]:.2f} '
                         f'open={p["open_iter"]} min_area={p["min_area"]} '
                         f'expand={p["expand"]:.2f}')
            if p['use_watershed']:
                param_str += f' dist_sigma={p["dist_sigma"]:.1f}'
            self.signals.log.emit(f'  Params: {param_str}')

            masks, binary, th_used, stats = segment_threshold(
                self._img_gray,
                method=p['method'],
                sigma=p['sigma'],
                th_coef=p['th_coef'],
                open_iter=p['open_iter'],
                min_area=p['min_area'],
                expand=p['expand'],
                use_watershed=p['use_watershed'],
                dist_sigma=p['dist_sigma'],
            )
            t_seg = time.time() - t0

            # Overlay
            overlay, label_rgb = make_colored_overlay(self._img_rgb, masks)
            t_overlay = time.time() - t0

            # Stats
            grains = compute_grain_stats(masks, self._img_rgb.shape)
            n = len(masks)
            total_area = sum(g['area'] for g in grains)
            area_pct = total_area / (self._img_rgb.shape[0] * self._img_rgb.shape[1]) * 100
            avg_diam = float(np.mean([g['diameter_px'] for g in grains])) if grains else 0
            t_total = time.time() - t0

            self.signals.log.emit(
                f'  ✅ Blocks: {n}  Area: {total_area} px ({area_pct:.1f}%)  '
                f'Avg Diam: {avg_diam:.1f} px'
            )
            self.signals.log.emit(
                f'  ⏱  Segment: {t_seg:.3f}s  Overlay: {t_overlay-t_seg:.3f}s  '
                f'Total: {t_total:.3f}s'
            )

            result = {
                'masks': masks,
                'binary': binary,
                'label_rgb': label_rgb,
                'overlay': overlay,
                'th_used': th_used,
                'stats': stats,
                'grains': grains,
                'n_blocks': n,
                'total_area': total_area,
                'area_pct': area_pct,
                'avg_diam': avg_diam,
                'time': t_total,
            }
            self.signals.finished.emit(result)

        except Exception as e:
            import traceback
            self.signals.error.emit(str(e))
            self.signals.log.emit(f'  ❌ Error: {e}')
            self.signals.log.emit(traceback.format_exc()[-500:])


# ====================================================================
#  GUI
# ====================================================================

class ImageWidget(QLabel):
    def __init__(self, title=''):
        super().__init__()
        self.title = title
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(280, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setStyleSheet('''
            QLabel {
                background-color: #1e1e1e;
                border: 1px solid #555;
                border-radius: 4px;
                color: #888;
                font-size: 13px;
            }
        ''')
        self.setText(title)
        self._pixmap = None

    def set_image(self, img_np):
        if img_np is None:
            self.setText(self.title)
            self._pixmap = None
            self.setPixmap(QPixmap())
            return

        h, w = img_np.shape[:2]
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
            img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        fmt = QImage.Format.Format_RGBA8888 if img_np.shape[2] == 4 else QImage.Format.Format_RGB888
        qimg = QImage(img_np.data, w, h, img_np.strides[0], fmt)
        self._pixmap = QPixmap.fromImage(qimg)
        self._update_pixmap()

    def _update_pixmap(self):
        if self._pixmap is not None:
            scaled = self._pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap()


class ParamSlider(QWidget):
    def __init__(self, name, min_val, max_val, default, step=0.01, fmt='{:.2f}'):
        super().__init__()
        self.fmt = fmt
        self._name = name
        self._fmt = fmt
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(f'{name} ({fmt.format(default)})')
        self.label.setFixedWidth(130)
        layout.addWidget(self.label)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(int(min_val / step))
        self.slider.setMaximum(int(max_val / step))
        self.slider.setValue(int(default / step))
        self.slider.valueChanged.connect(self._on_change)
        layout.addWidget(self.slider, 1)
        self._step = step

    def _on_change(self, val):
        self.label.setText(f'{self._name} ({self._fmt.format(val * self._step)})')

    def value(self):
        return self.slider.value() * self._step

    def setValue(self, v):
        self.slider.setValue(int(v / self._step))


class IntParamSlider(QWidget):
    def __init__(self, name, min_val, max_val, default):
        super().__init__()
        self._name = name
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(f'{name} ({default})')
        self.label.setFixedWidth(130)
        layout.addWidget(self.label)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(min_val)
        self.slider.setMaximum(max_val)
        self.slider.setValue(default)
        self.slider.valueChanged.connect(self._on_change)
        layout.addWidget(self.slider, 1)

    def _on_change(self, v):
        self.label.setText(f'{self._name} ({v})')

    def value(self):
        return self.slider.value()

    def setValue(self, v):
        self.slider.setValue(v)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('DD5 Alloy Segmentation — 阈值调参工具')
        self.setMinimumSize(1300, 850)

        self._img = None
        self._img_gray = None
        self._masks = []
        self._grains = []
        self._image_path = None
        self._worker = None

        self._setup_ui()
        self._setup_menu()
        pass  # auto-run disabled, only Run button triggers processing

    def _setup_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu('File')
        a = QAction('Open Image...', self)
        a.setShortcut('Ctrl+O'); a.triggered.connect(self._open_image); fm.addAction(a)
        a = QAction('Save Results...', self)
        a.setShortcut('Ctrl+S'); a.triggered.connect(self._save_results); fm.addAction(a)
        fm.addSeparator()
        a = QAction('Quit', self)
        a.setShortcut('Ctrl+Q'); a.triggered.connect(self.close); fm.addAction(a)

        vm = mb.addMenu('View')
        a = QAction('Clear Log', self)
        a.triggered.connect(lambda: self.log.clear()); vm.addAction(a)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        ml = QVBoxLayout(central)
        ml.setSpacing(6)

        # === Toolbar ===
        tb = QHBoxLayout()
        self.btn_open = QPushButton('\U0001F4C2 Open')
        self.btn_open.clicked.connect(self._open_image)
        tb.addWidget(self.btn_open)

        self.btn_save = QPushButton('\U0001F4BE Save')
        self.btn_save.clicked.connect(self._save_results)
        self.btn_save.setEnabled(False)
        tb.addWidget(self.btn_save)

        tb.addWidget(QLabel('  Method:'))
        self.cb_method = QComboBox()
        for m in ['Otsu', 'Otsu strict (x1.10)', 'Otsu very strict (x1.20)',
                   'Yen', 'Triangle', 'Li', 'Isodata', 'Mean', 'Minimum',
                   'Multi-Otsu', '---', 'Gradient Watershed']:
            self.cb_method.addItem(m)
        self.cb_method.currentIndexChanged.connect(self._on_param_changed_noauto)
        tb.addWidget(self.cb_method)

        self.btn_run = QPushButton('▶ Run')
        self.btn_run.clicked.connect(self._start_worker)
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet(
            'QPushButton { background:#4CAF50; color:white; font-weight:bold;'
            ' padding:6px 18px; border-radius:4px; }'
            'QPushButton:hover { background:#45a049; }'
            'QPushButton:disabled { background:#888; }')
        tb.addWidget(self.btn_run)
        tb.addStretch()
        ml.addLayout(tb)

        # === Image displays ===
        imgs = QHBoxLayout()
        self.v0 = ImageWidget('Original\n(load an image)')
        self.v1 = ImageWidget('Binary Mask')
        self.v2 = ImageWidget('Segmentation Result')
        imgs.addWidget(self.v0)
        imgs.addWidget(self.v1)
        imgs.addWidget(self.v2)
        ml.addLayout(imgs, 1)

        # === Parameters (bottom) ===
        pg = QGroupBox('Parameters')
        pgrid = QGridLayout(pg)
        pgrid.setSpacing(3)

        self.p_sigma = ParamSlider('Sigma', 0, 3.0, 0.8, 0.05, '{:.2f}')
        self.p_sigma.slider.valueChanged.connect(self._on_param_changed_noauto)
        pgrid.addWidget(QLabel('Gaussian blur'), 0, 0)
        pgrid.addWidget(self.p_sigma, 0, 1)

        self.p_th_coef = ParamSlider('TH Coef', 0.5, 1.5, 1.0, 0.01)
        self.p_th_coef.slider.valueChanged.connect(self._on_param_changed_noauto)
        pgrid.addWidget(QLabel('Threshold coef'), 0, 2)
        pgrid.addWidget(self.p_th_coef, 0, 3)

        self.p_expand = ParamSlider('Expand', 0, 0.5, 0.10, 0.01)
        self.p_expand.slider.valueChanged.connect(self._on_param_changed_noauto)
        pgrid.addWidget(QLabel('Box expand'), 0, 4)
        pgrid.addWidget(self.p_expand, 0, 5)

        self.p_open = IntParamSlider('Open Iter', 0, 10, 1)
        self.p_open.slider.valueChanged.connect(self._on_param_changed_noauto)
        pgrid.addWidget(QLabel('Opening'), 1, 0)
        pgrid.addWidget(self.p_open, 1, 1)

        self.p_min_area = IntParamSlider('Min Area', 1, 500, 30)
        self.p_min_area.slider.valueChanged.connect(self._on_param_changed_noauto)
        pgrid.addWidget(QLabel('Min area'), 1, 2)
        pgrid.addWidget(self.p_min_area, 1, 3)

        self.p_dist_sigma = ParamSlider('Dist Sigma', 0.5, 10.0, 3.0, 0.1, '{:.1f}')
        self.p_dist_sigma.slider.valueChanged.connect(self._on_param_changed_noauto)
        pgrid.addWidget(QLabel('Dist sigma'), 1, 4)
        pgrid.addWidget(self.p_dist_sigma, 1, 5)

        ml.addWidget(pg)

        # === Log panel ===
        dock = QDockWidget('Log', self)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont('Consolas', 10))
        self.log.setMaximumBlockCount(500)
        self.log.setStyleSheet('QTextEdit { background:#1a1a2e; color:#e0e0e0; }')
        dock.setWidget(self.log)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

        self.log_msg(
            'DD5 Alloy Segmentation — 阈值调参工具 loaded.\n'
            'Open an image (Ctrl+O) to begin.\n'
            'Adjust parameters and results update automatically.\n')

        # === Status ===
        self.sb = QStatusBar()
        self.setStatusBar(self.sb)
        self.sb_label = QLabel('Ready')
        self.sb.addWidget(self.sb_label)

    def log_msg(self, msg):
        self.log.appendPlainText(msg.strip())

    def _on_param_changed_noauto(self):
        pass  # parameter changed, but only Run button triggers processing

    def _is_busy(self):
        return self._worker is not None and self._worker.isRunning()

    def _selected_params(self):
        t = self.cb_method.currentText()
        if 'Watershed' in t:
            return {'method': 'otsu', 'use_watershed': True,
                    'th_coef': self.p_th_coef.value()}
        if 'x1.10' in t:
            return {'method': 'otsu', 'use_watershed': False, 'th_coef': 1.10}
        if 'x1.20' in t:
            return {'method': 'otsu', 'use_watershed': False, 'th_coef': 1.20}
        m = t.lower().replace('-', '')
        if m in THRESHOLD_METHODS:
            return {'method': m, 'use_watershed': False,
                    'th_coef': self.p_th_coef.value()}
        return {'method': 'otsu', 'use_watershed': False,
                'th_coef': self.p_th_coef.value()}

    def _start_worker(self):
        if self._img is None:
            return
        if self._is_busy():
            self.log_msg('  ⏳ Worker busy, skipping...')
            return

        p = self._selected_params()
        params = {
            'method': p['method'],
            'sigma': self.p_sigma.value(),
            'th_coef': p['th_coef'],
            'open_iter': self.p_open.value(),
            'min_area': self.p_min_area.value(),
            'expand': self.p_expand.value(),
            'use_watershed': p['use_watershed'],
            'dist_sigma': self.p_dist_sigma.value(),
        }

        self.btn_run.setEnabled(False)
        self.btn_run.setText('⏳ Running...')

        self._worker = SegmentationWorker(self._img_gray, self._img, params)
        self._worker.signals.log.connect(self.log_msg)
        self._worker.signals.error.connect(self._on_worker_error)
        self._worker.signals.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_worker_finished(self, result):
        self._masks = result['masks']
        self._grains = result['grains']
        self.v1.set_image(result['binary'].astype(np.float32))
        self.v2.set_image(result['label_rgb'])

        n = result['n_blocks']
        grains = result['grains']
        if grains:
            avg_major = float(np.mean([g['major_axis_px'] for g in grains]))
            avg_minor = float(np.mean([g['minor_axis_px'] for g in grains]))
        else:
            avg_major = avg_minor = 0.0
        self.sb_label.setText(
            f'Blocks: {n}  Avg: {avg_major:.1f}×{avg_minor:.1f} px  '
            f'Diam: {result["avg_diam"]:.1f} px  '
            f'Area: {result["total_area"]} px ({result["area_pct"]:.1f}%)  '
            f'Image: {self._img.shape[1]}×{self._img.shape[0]}  '
            f'Time: {result["time"]:.3f}s'
        )

        self.btn_run.setEnabled(True)
        self.btn_run.setText('▶ Run')
        self._worker = None

    def _on_worker_error(self, msg):
        self.log_msg(f'  ❌ Error: {msg}')
        self.sb_label.setText(f'Error: {msg}')
        self.btn_run.setEnabled(True)
        self.btn_run.setText('▶ Run')
        self._worker = None

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Image', '',
            'Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)')
        if not path:
            return
        self._load_image(path)

    def _load_image(self, path):
        try:
            img = Image.open(path).convert('RGB')
            self._img = np.array(img)
            self._img_gray = np.mean(self._img, axis=2)
            self._image_path = path
            self._masks = []
            self._grains = []

            self.v0.set_image(self._img)
            self.v1.set_image(None)
            self.v2.set_image(None)
            self.btn_run.setEnabled(True)
            self.btn_save.setEnabled(True)

            fn = Path(path).name
            h, w = self._img.shape[:2]
            self.log_msg(
                f'[{datetime.now():%H:%M:%S}] \U0001F4C4 Loaded: {fn}  {w}×{h}  '
                f'gray: {self._img_gray.min():.0f}-{self._img_gray.max():.0f}  '
                f'mean: {self._img_gray.mean():.0f}'
            )
            self.sb_label.setText(f'{fn}  {w}x{h}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed: {e}')

    def _save_results(self):
        if not self._masks:
            QMessageBox.information(self, 'Info', 'No results.')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Results', 'results.json',
            'JSON (*.json);;PNG Image (*.png)')
        if not path:
            return
        ext = Path(path).suffix.lower()
        try:
            if ext == '.json':
                out = {
                    'image': Path(self._image_path).name if self._image_path else '',
                    'n_grains': len(self._grains),
                    'grains': self._grains,
                }
                with open(path, 'w') as f:
                    json.dump(out, f, indent=2, ensure_ascii=False)
                self.log_msg(f'[{datetime.now():%H:%M:%S}] \U0001F4BE Saved: {path}')
            elif ext == '.png':
                _, label_rgb = make_colored_overlay(self._img, self._masks)
                Image.fromarray(label_rgb).save(path)
                self.log_msg(f'[{datetime.now():%H:%M:%S}] \U0001F4BE Saved: {path}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Save failed: {e}')


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
