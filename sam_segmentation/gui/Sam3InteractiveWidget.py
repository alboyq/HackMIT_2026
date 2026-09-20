import io
import os

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


def _build_image_filters():
    """Build file-dialog filter strings from every format PIL can open."""
    exts = sorted(
        ext for ext, fmt in PIL.Image.registered_extensions().items()
        if fmt in PIL.Image.OPEN
    )
    # Windows PowerShell filter: semicolon-separated
    ps_exts = ";".join(f"*{e}" for e in exts)
    ps_filter = f"Image files|{ps_exts}|All files|*.*"
    # Qt filter: space-separated
    qt_exts = " ".join(f"*{e}" for e in exts)
    qt_filter = f"Image Files ({qt_exts})"
    return ps_filter, qt_filter


_PS_FILTER, _QT_FILTER = _build_image_filters()
import requests

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.patches import Rectangle

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QCursor, QIcon, QFont
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QSlider, QRadioButton,
    QButtonGroup, QGroupBox, QFileDialog, QFrame,
    QSizePolicy, QScrollArea, QApplication,
    QGraphicsDropShadowEffect,
)


# ── Dark-theme stylesheet ───────────────────────────────────────────────
STYLESHEET = """
QMainWindow {
    background-color: #12131e;
}

/* ── Sidebar card ─────────────────────────────────── */
#Sidebar {
    background-color: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 #1e2035, stop:1 #1a1b2e
    );
    border: 1px solid rgba(108, 99, 255, 0.15);
    border-radius: 16px;
}

/* ── Group boxes ──────────────────────────────────── */
QGroupBox {
    background-color: rgba(30, 32, 53, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    margin-top: 18px;
    padding: 16px 12px 12px 12px;
    font-size: 11px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.45);
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 10px;
    color: rgba(255, 255, 255, 0.45);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Accent (primary) button ──────────────────────── */
QPushButton#AccentBtn {
    background-color: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #6c63ff, stop:1 #b06ab3
    );
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 9px 18px;
    font-size: 13px;
    font-weight: 600;
}
QPushButton#AccentBtn:hover {
    background-color: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #7e77ff, stop:1 #c07ec5
    );
}
QPushButton#AccentBtn:pressed {
    background-color: #5a52e0;
}
QPushButton#AccentBtn:disabled {
    background-color: #3a3b55;
    color: #6b6c80;
}

/* ── Quiet (secondary) button ─────────────────────── */
QPushButton#QuietBtn {
    background-color: rgba(255, 255, 255, 0.06);
    color: #c0c1d4;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 12px;
    font-weight: 500;
}
QPushButton#QuietBtn:hover {
    background-color: rgba(255, 255, 255, 0.10);
    border-color: rgba(108, 99, 255, 0.4);
    color: #e0e1f0;
}
QPushButton#QuietBtn:pressed {
    background-color: rgba(108, 99, 255, 0.18);
}
QPushButton#QuietBtn:disabled {
    color: #4a4b60;
    border-color: rgba(255, 255, 255, 0.04);
}

/* ── Line edits ───────────────────────────────────── */
QLineEdit {
    background-color: rgba(0, 0, 0, 0.25);
    color: #e0e1f0;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
    selection-background-color: #6c63ff;
}
QLineEdit:focus {
    border-color: #6c63ff;
}

/* ── Radio buttons ────────────────────────────────── */
QRadioButton {
    color: #c0c1d4;
    font-size: 12px;
    spacing: 6px;
}
QRadioButton::indicator {
    width: 16px; height: 16px;
    border-radius: 8px;
    border: 2px solid rgba(255, 255, 255, 0.20);
    background-color: transparent;
}
QRadioButton::indicator:checked {
    background-color: #6c63ff;
    border-color: #6c63ff;
}
QRadioButton::indicator:hover {
    border-color: rgba(108, 99, 255, 0.6);
}

/* ── Sliders ──────────────────────────────────────── */
QSlider::groove:horizontal {
    height: 6px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:1,
        stop:0 #6c63ff, stop:1 #b06ab3
    );
    width: 16px; height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: #7e77ff;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #6c63ff, stop:1 #b06ab3
    );
    border-radius: 3px;
}

/* ── Labels ───────────────────────────────────────── */
QLabel {
    color: #c0c1d4;
    font-size: 12px;
}
QLabel#TitleLabel {
    font-size: 20px;
    font-weight: 700;
    color: #ffffff;
}
QLabel#SubtitleLabel {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.45);
}
QLabel#StatusLabel {
    font-size: 12px;
    color: #9e9fb8;
    padding: 4px 0;
}
QLabel#ValueLabel {
    font-size: 11px;
    color: #9e9fb8;
}
QLabel#SliderLabel {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.55);
}

/* ── Scroll area ──────────────────────────────────── */
QScrollArea {
    background: transparent;
    border: none;
}
QScrollArea > QWidget > QWidget {
    background: transparent;
}
QScrollBar:vertical {
    background: transparent;
    width: 6px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: rgba(255, 255, 255, 0.12);
    border-radius: 3px;
    min-height: 30px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

/* ── Canvas frame ─────────────────────────────────── */
#CanvasFrame {
    background-color: #0d0e18;
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 12px;
}
"""


class Sam3DesktopApp(QMainWindow):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor

        # ── state ──
        self.state = None
        self.current_image = None
        self.current_image_array = None
        self.box_mode = "positive"
        self.drawing_box = False
        self.box_start = None
        self.current_rect = None

        self.setWindowTitle("SAM3 Interactive Segmentation")
        self.resize(1520, 960)
        self.setMinimumSize(QSize(1100, 700))
        self.setStyleSheet(STYLESHEET)

        self._build_ui()
        self._setup_plot()

    # ────────────────────────────────────────────────────────
    #  UI construction
    # ────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(16)

        # ── Sidebar ──────────────────────────────────────
        sidebar_card = QFrame()
        sidebar_card.setObjectName("Sidebar")
        sidebar_card.setFixedWidth(360)
        sidebar_card.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(40)
        shadow.setOffset(0, 4)
        shadow.setColor(Qt.GlobalColor.black)
        sidebar_card.setGraphicsEffect(shadow)

        sidebar_inner = QVBoxLayout(sidebar_card)
        sidebar_inner.setContentsMargins(20, 24, 20, 20)
        sidebar_inner.setSpacing(6)

        # Title
        title = QLabel("SAM3 Interactive\nSegmentation")
        title.setObjectName("TitleLabel")
        title.setWordWrap(True)
        sidebar_inner.addWidget(title)

        subtitle = QLabel("Text prompts  ·  Positive / negative box prompts")
        subtitle.setObjectName("SubtitleLabel")
        sidebar_inner.addWidget(subtitle)

        # Status
        self.status_label = QLabel("Load an image to begin")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setWordWrap(True)
        sidebar_inner.addWidget(self.status_label)
        sidebar_inner.addSpacing(4)

        # ── Scrollable controls area ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_content = QWidget()
        controls_layout = QVBoxLayout(scroll_content)
        controls_layout.setContentsMargins(0, 0, 4, 0)
        controls_layout.setSpacing(12)
        scroll.setWidget(scroll_content)

        # ── Image Source group ───────────────────────────
        img_group = QGroupBox("IMAGE SOURCE")
        img_lay = QVBoxLayout(img_group)
        img_lay.setSpacing(8)

        self.load_button = QPushButton("  Open Image…")
        self.load_button.setObjectName("AccentBtn")
        self.load_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.load_button.clicked.connect(self._on_load_file)
        img_lay.addWidget(self.load_button)

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Paste image URL…")
        img_lay.addWidget(self.url_input)

        self.url_button = QPushButton("Load URL")
        self.url_button.setObjectName("QuietBtn")
        self.url_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.url_button.clicked.connect(self._on_load_url)
        img_lay.addWidget(self.url_button)

        controls_layout.addWidget(img_group)

        # ── Segmentation group ───────────────────────────
        seg_group = QGroupBox("SEGMENTATION")
        seg_lay = QVBoxLayout(seg_group)
        seg_lay.setSpacing(8)

        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText('Text prompt  (e.g. "person", "dog")')
        self.prompt_input.returnPressed.connect(self._on_text_prompt)
        seg_lay.addWidget(self.prompt_input)

        self.segment_button = QPushButton("  Segment")
        self.segment_button.setObjectName("AccentBtn")
        self.segment_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.segment_button.clicked.connect(self._on_text_prompt)
        seg_lay.addWidget(self.segment_button)

        # Box mode radios
        mode_label = QLabel("Box Mode")
        mode_label.setObjectName("SliderLabel")
        seg_lay.addWidget(mode_label)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(16)
        self.radio_positive = QRadioButton("Positive")
        self.radio_negative = QRadioButton("Negative")
        self.radio_positive.setChecked(True)
        self.box_mode_group = QButtonGroup(self)
        self.box_mode_group.addButton(self.radio_positive)
        self.box_mode_group.addButton(self.radio_negative)
        self.box_mode_group.buttonClicked.connect(self._on_box_mode_change)
        mode_row.addWidget(self.radio_positive)
        mode_row.addWidget(self.radio_negative)
        mode_row.addStretch()
        seg_lay.addLayout(mode_row)

        # Confidence slider
        conf_header = QHBoxLayout()
        conf_lbl = QLabel("Confidence")
        conf_lbl.setObjectName("SliderLabel")
        self.conf_value_lbl = QLabel("0.50")
        self.conf_value_lbl.setObjectName("ValueLabel")
        conf_header.addWidget(conf_lbl)
        conf_header.addStretch()
        conf_header.addWidget(self.conf_value_lbl)
        seg_lay.addLayout(conf_header)

        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self._on_confidence_change)
        seg_lay.addWidget(self.confidence_slider)

        seg_lay.addSpacing(4)

        self.clear_button = QPushButton("Clear All Prompts")
        self.clear_button.setObjectName("QuietBtn")
        self.clear_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.clear_button.clicked.connect(self._on_clear_prompts)
        seg_lay.addWidget(self.clear_button)

        controls_layout.addWidget(seg_group)

        # ── Display group ────────────────────────────────
        disp_group = QGroupBox("DISPLAY")
        disp_lay = QVBoxLayout(disp_group)
        disp_lay.setSpacing(8)

        dw_header = QHBoxLayout()
        dw_lbl = QLabel("Display Width")
        dw_lbl.setObjectName("SliderLabel")
        self.dw_value_lbl = QLabel("960 px")
        self.dw_value_lbl.setObjectName("ValueLabel")
        dw_header.addWidget(dw_lbl)
        dw_header.addStretch()
        dw_header.addWidget(self.dw_value_lbl)
        disp_lay.addLayout(dw_header)

        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(300, 2000)
        self.size_slider.setValue(960)
        self.size_slider.valueChanged.connect(self._on_size_change)
        disp_lay.addWidget(self.size_slider)

        controls_layout.addWidget(disp_group)
        controls_layout.addStretch()

        sidebar_inner.addWidget(scroll, 1)
        root_layout.addWidget(sidebar_card)

        # ── Canvas area ──────────────────────────────────
        canvas_frame = QFrame()
        canvas_frame.setObjectName("CanvasFrame")
        self.canvas_layout = QVBoxLayout(canvas_frame)
        self.canvas_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.addWidget(canvas_frame, 1)

        # Track interactive controls for enable/disable
        self._interactive_controls = [
            self.load_button, self.url_button,
            self.segment_button, self.clear_button,
            self.confidence_slider, self.size_slider,
        ]

    # ────────────────────────────────────────────────────────
    #  Matplotlib setup
    # ────────────────────────────────────────────────────────
    def _setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.patch.set_facecolor("#0d0e18")
        self.ax.axis("off")
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas_layout.addWidget(self.canvas)

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

    # ────────────────────────────────────────────────────────
    #  Loading state
    # ────────────────────────────────────────────────────────
    def _set_loading(self, is_loading, message="Processing…"):
        self.status_label.setText(f"⏳  {message}" if is_loading else message)
        for ctrl in self._interactive_controls:
            ctrl.setEnabled(not is_loading)
        QApplication.processEvents()

    # ────────────────────────────────────────────────────────
    #  Image loading
    # ────────────────────────────────────────────────────────
    @staticmethod
    def _is_wsl():
        try:
            with open("/proc/version", "r") as f:
                return "microsoft" in f.read().lower()
        except OSError:
            return False

    def _on_load_file(self):
        import subprocess

        path = None

        if self._is_wsl():
            # Use the real Windows native file dialog via PowerShell
            ps_script = (
                "Add-Type -AssemblyName System.Windows.Forms;"
                "$d = New-Object System.Windows.Forms.OpenFileDialog;"
                "$d.Title = 'Select Image';"
                f"$d.Filter = '{_PS_FILTER}';"
                "if ($d.ShowDialog() -eq 'OK') { $d.FileName } else { '' }"
            )
            try:
                result = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-Command", ps_script],
                    capture_output=True, text=True, timeout=120,
                )
                win_path = result.stdout.strip()
                if win_path:
                    # Convert Windows path → WSL path
                    wsl = subprocess.run(
                        ["wslpath", "-u", win_path],
                        capture_output=True, text=True,
                    )
                    path = wsl.stdout.strip()
            except Exception as exc:
                self.status_label.setText(f"❌  File dialog error: {exc}")
                return
        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Image",
                os.path.expanduser("~"),
                _QT_FILTER,
            )

        if not path:
            return
        image = PIL.Image.open(path).convert("RGB")
        self._set_image(image)

    def _on_load_url(self):
        url = self.url_input.text().strip()
        if not url:
            self.status_label.setText("Please enter an image URL")
            return
        self._set_loading(True, "Downloading image from URL…")
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            image = PIL.Image.open(io.BytesIO(response.content)).convert("RGB")
            self._set_image(image)
        except Exception as exc:
            self._set_loading(False, f"❌  Error loading URL: {exc}")

    def _set_image(self, image):
        self._set_loading(True, "Running image encoder…")
        try:
            self.current_image = image
            self.current_image_array = np.array(image)
            self.state = self.processor.set_image(image)
            self._resize_figure()
            self._update_display()
            self._set_loading(False, f"✅  Image loaded: {image.size[0]}×{image.size[1]}")
        except Exception as exc:
            self._set_loading(False, f"❌  Error processing image: {exc}")

    # ────────────────────────────────────────────────────────
    #  Prompts
    # ────────────────────────────────────────────────────────
    def _on_text_prompt(self):
        if self.state is None:
            self.status_label.setText("Please load an image first")
            return
        prompt = self.prompt_input.text().strip()
        if not prompt:
            self.status_label.setText("Please enter a text prompt")
            return
        self._set_loading(True, f'Segmenting with prompt: "{prompt}"')
        try:
            self.state = self.processor.set_text_prompt(prompt, self.state)
            self._update_display()
            self._set_loading(False, f'✅  Segmented: "{prompt}"')
        except Exception as exc:
            self._set_loading(False, f"❌  Error segmenting: {exc}")

    def _on_box_mode_change(self, button):
        self.box_mode = "positive" if button is self.radio_positive else "negative"

    def _on_clear_prompts(self):
        if self.state is None:
            return
        self._set_loading(True, "Clearing prompts…")
        try:
            self.state = self.processor.reset_all_prompts(self.state)
            if "prompted_boxes" in self.state:
                del self.state["prompted_boxes"]
            self.prompt_input.clear()
            self._update_display()
            self._set_loading(False, "✅  Cleared all prompts")
        except Exception as exc:
            self._set_loading(False, f"❌  Error clearing prompts: {exc}")

    def _on_confidence_change(self, value):
        conf = value / 100.0
        self.conf_value_lbl.setText(f"{conf:.2f}")
        if self.state is None:
            return
        try:
            self.state = self.processor.set_confidence_threshold(conf, self.state)
            self._update_display()
            self.status_label.setText("Updated confidence threshold")
        except Exception as exc:
            self.status_label.setText(f"❌  Error updating confidence: {exc}")

    # ────────────────────────────────────────────────────────
    #  Display / resize
    # ────────────────────────────────────────────────────────
    def _resize_figure(self):
        if self.current_image is None:
            return
        img_w, img_h = self.current_image.size
        display_w = float(self.size_slider.value())
        display_h = int(display_w * (img_h / img_w))
        dpi = self.fig.dpi
        self.fig.set_size_inches(display_w / dpi, display_h / dpi, forward=True)

    def _on_size_change(self, value):
        self.dw_value_lbl.setText(f"{value} px")
        if self.current_image is None:
            return
        self._resize_figure()
        self._update_display()

    # ────────────────────────────────────────────────────────
    #  Box drawing (matplotlib events)
    # ────────────────────────────────────────────────────────
    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self.drawing_box = True
        self.box_start = (event.xdata, event.ydata)

    def _on_motion(self, event):
        if (
            not self.drawing_box
            or event.inaxes != self.ax
            or self.box_start is None
            or event.xdata is None
            or event.ydata is None
        ):
            return
        if self.current_rect is not None:
            self.current_rect.remove()

        x0, y0 = self.box_start
        x1, y1 = event.xdata, event.ydata
        color = "green" if self.box_mode == "positive" else "red"
        self.current_rect = Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor=color, linewidth=2, linestyle="--",
        )
        self.ax.add_patch(self.current_rect)
        self.canvas.draw_idle()

    def _on_release(self, event):
        if (
            not self.drawing_box
            or event.inaxes != self.ax
            or self.box_start is None
            or event.xdata is None
            or event.ydata is None
            or self.state is None
        ):
            self.drawing_box = False
            self.box_start = None
            return

        self.drawing_box = False
        if self.current_rect is not None:
            self.current_rect.remove()
            self.current_rect = None

        x0, y0 = self.box_start
        x1, y1 = event.xdata, event.ydata
        self.box_start = None

        x_min, x_max = min(x0, x1), max(x0, x1)
        y_min, y_max = min(y0, y1), max(y0, y1)
        if abs(x_max - x_min) < 5 or abs(y_max - y_min) < 5:
            self._update_display()
            return

        img_h = self.state["original_height"]
        img_w = self.state["original_width"]
        box = [
            (x_min + x_max) / 2.0 / img_w,
            (y_min + y_max) / 2.0 / img_h,
            (x_max - x_min) / img_w,
            (y_max - y_min) / img_h,
        ]
        label = self.box_mode == "positive"
        mode_str = "positive" if label else "negative"

        if "prompted_boxes" not in self.state:
            self.state["prompted_boxes"] = []
        self.state["prompted_boxes"].append(
            {"box": [x_min, y_min, x_max, y_max], "label": label}
        )

        self._set_loading(True, f"Adding {mode_str} box…")
        try:
            self.state = self.processor.add_geometric_prompt(box, label, self.state)
            self._update_display()
            self._set_loading(False, f"✅  Added {mode_str} box")
        except Exception as exc:
            self._set_loading(False, f"❌  Error adding {mode_str} box: {exc}")

    # ────────────────────────────────────────────────────────
    #  Render
    # ────────────────────────────────────────────────────────
    def _update_display(self):
        if self.current_image_array is None:
            return

        self.ax.clear()
        self.ax.axis("off")
        self.ax.imshow(self.current_image_array)

        if self.state is not None and "masks" in self.state:
            masks = self.state.get("masks", [])
            boxes = self.state.get("boxes", [])
            scores = self.state.get("scores", [])

            if len(masks) > 0:
                overlay = np.zeros((*self.current_image_array.shape[:2], 4))
                for idx, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
                    mask_np = mask[0].cpu().numpy()
                    color = plt.cm.tab10(idx % 10)[:3]
                    overlay[mask_np > 0.5] = (*color, 0.5)

                    x0, y0, x1, y1 = box.cpu().numpy()
                    rect = Rectangle(
                        (x0, y0), x1 - x0, y1 - y0,
                        fill=False, edgecolor=color, linewidth=2,
                    )
                    self.ax.add_patch(rect)
                    self.ax.text(
                        x0, y0 - 5, f"{float(score):.2f}",
                        color="white", fontsize=10,
                        bbox={"facecolor": color, "alpha": 0.7, "edgecolor": "none", "pad": 2},
                    )
                self.ax.imshow(overlay)

        if self.state is not None and "prompted_boxes" in self.state:
            for prompted_box in self.state["prompted_boxes"]:
                x0, y0, x1, y1 = prompted_box["box"]
                color = "green" if prompted_box["label"] else "red"
                rect = Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    fill=False, edgecolor=color, linewidth=2, linestyle="--",
                )
                self.ax.add_patch(rect)

        self.canvas.draw_idle()