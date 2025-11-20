#!/usr/bin/env python3
"""
Multi-camera GUI

Supports:
- Real cameras via Harvester + your HarvesterCameraManager backend.
- Simulated cameras for development / testing.
- A grid of tiles that shows all cameras (real + simulated).
- Click-to-select an active camera.
- Settings/Calibration docks.

Settings currently support:
- Per-camera name (persistent, keyed by hardware ID)
- Gain (real cameras only)
- Exposure (real cameras only)
- Preview FPS (real + simulated)
- Global recording settings (all cameras):
    - Save directory
    - Session ID
    - Record video on/off
    - Snapshot FPS (per second)
    - Recording FPS (video_fps)
    - Video codec (XVID/MJPG/MP4V)

Recording:
- Real cameras: via backend.HarvesterCameraManager.add_file()/start()/stop().
- Simulated cameras: handled here with cv2.VideoWriter + cv2.imwrite.
"""

import sys
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QToolBar,
    QAction,
    QStatusBar,
    QDockWidget,
    QLineEdit,
    QPushButton,
    QFormLayout,
    QDoubleSpinBox,
    QSizePolicy,
    QFileDialog,
    QCheckBox,
    QGroupBox,
    QComboBox,
)

from backend import HarvesterCameraManager
from calibration_panel import CalibrationPanel
from ptv import PTVTracker, PTVConfig

try:
    from harvesters.core import Harvester
except ImportError:  # pragma: no cover
    Harvester = None


# ---------------------------------------------------------------------------
# CONFIG: tweak these for your setup
# ---------------------------------------------------------------------------

# CTI path for real cameras
CTI_PATH = "VimbaGigETL.cti"

# Enable/disable real cameras
ENABLE_REAL_CAMERAS = True

# Enable simulation and how many simulated cameras to add
ENABLE_SIMULATED_CAMERAS = False
NUM_SIMULATED_CAMERAS = 5  # e.g. 1 real + 7 sim = 8 total

# Default preview FPS for real vs simulated cameras
PREVIEW_FPS_REAL = 2.0
PREVIEW_FPS_SIM = 5.0

# Target aspect ratio for a camera tile (width / height)
TARGET_TILE_ASPECT = 5 / 4  # good match for 640x480

# Where to store persistent camera names (per hardware ID)
CONFIG_PATH = Path.cwd() / ".multicam_harvester_gui.json"
print(f"CONFIG_PATH: {CONFIG_PATH}")

# ---------------------------------------------------------------------------
# Simulated backend for a single camera
# ---------------------------------------------------------------------------

class SimulatedCameraBackend:
    """
    Very simple "camera" that generates synthetic frames.

    Exposes:
      - streaming (bool)
      - start_streaming()
      - stop_streaming()
      - fetch_next(timeout) -> np.ndarray | None
    """

    def __init__(self, camera_id: str, name: str, fps: float = 5.0):
        self.camera_id = camera_id
        self.name = name
        self.preview_fps = float(fps)
        self.streaming = False
        self._frame_idx = 0

    def start_streaming(self):
        self.streaming = True

    def stop_streaming(self):
        self.streaming = False

    # For compatibility with HarvesterCameraManager
    def fetch_next(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        if not self.streaming:
            return None
        return self._get_next_frame()

    def _get_next_frame(self) -> np.ndarray:
        h, w = 480, 640
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Simple colored background based on camera id hash
        base = hash(self.camera_id) & 0xFFFFFF
        b = 150 + (base & 0x3F)
        g = 150 + ((base >> 6) & 0x3F)
        r = 150 + ((base >> 12) & 0x3F)
        img[:, :] = (b, g, r)

        # Moving bar
        bar_width = 40
        x = int((self._frame_idx * 10) % (w + bar_width)) - bar_width
        x0 = max(0, x)
        x1 = min(w, x + bar_width)
        if x0 < x1:
            cv2.rectangle(img, (x0, 0), (x1, h), (255, 255, 255), thickness=-1)

        # Text overlay
        t = time.strftime("%H:%M:%S")
        cv2.putText(
            img,
            f"{self.name} ({self.camera_id})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"Frame {self._frame_idx}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            t,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        self._frame_idx += 1
        return img


# ---------------------------------------------------------------------------
# Grab thread: one per camera (real or simulated)
# ---------------------------------------------------------------------------

class GrabThread(QThread):
    frame_ready = pyqtSignal(str, np.ndarray)  # camera_id, frame

    def __init__(self, camera_id: str, backend: Any, preview_fps: float = 2.0, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self._backend = backend
        self._running = False
        self._preview_fps = float(preview_fps)

    def set_preview_fps(self, fps: float):
        """Update preview FPS from the GUI thread."""
        self._preview_fps = max(0.1, float(fps))  # avoid division by zero

    def run(self):
        # Same pattern as single-cam GUI:
        # - fetch_next() runs as fast as frames are available
        # - preview FPS only controls how often we emit to the GUI
        self._running = True
        last_emit = 0.0

        while self._running:
            if not getattr(self._backend, "streaming", False):
                self.msleep(5)
                continue

            frame = self._backend.fetch_next(timeout=0.1)
            if frame is None:
                self.msleep(5)
                continue

            now = time.time()
            target_dt = 1.0 / max(0.1, float(self._preview_fps))
            if now - last_emit >= target_dt:
                last_emit = now
                self.frame_ready.emit(self.camera_id, frame)

    def stop(self):
        self._running = False
        self.wait()



# ---------------------------------------------------------------------------
# UI: single camera tile
# ---------------------------------------------------------------------------

class CameraTile(QWidget):
    """
    One tile in the camera wall.

    Contains:
      - Name label
      - Image label
      - Status label
    """

    clicked = pyqtSignal(str)  # camera_id

    def __init__(self, camera_id: str, name: str, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.name = name
        self._active = False

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Name / header
        self.name_label = QLabel(self.name)
        self.name_label.setStyleSheet("color: #222222; font-weight: bold;")
        layout.addWidget(self.name_label)

        # Image:
        # - minimum size to avoid tiny tiles
        # - size policy Ignored so pixmap size doesn't drive layout geometry
        self.image_label = QLabel("No signal")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Ignored,
        )
        self.image_label.setStyleSheet("background: #e0e0e0; color: #666666;")
        layout.addWidget(self.image_label, 1)

        # Status line
        self.status_label = QLabel("Disconnected")
        self.status_label.setStyleSheet("color: #555555;")
        layout.addWidget(self.status_label)

        self._apply_active_style()

    def set_status(self, text: str):
        self.status_label.setText(text)

    def set_active(self, active: bool):
        self._active = active
        self._apply_active_style()

    def _apply_active_style(self):
        if self._active:
            self.setStyleSheet(
                "QWidget { border: 2px solid #0078D7; background: #ffffff; }"
            )
        else:
            self.setStyleSheet(
                "QWidget { border: 1px solid #cccccc; background: #f8f8f8; }"
            )

    def update_frame(self, frame: np.ndarray):
        """Convert np.ndarray (BGR or gray) to QPixmap and show, scaled to label's contents rect."""
        if frame.ndim == 2:
            disp = frame
            h, w = disp.shape
            bytes_per_line = w
            qimg = QImage(
                disp.data, w, h, bytes_per_line, QImage.Format_Grayscale8
            )
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(
                rgb.data, w, h, bytes_per_line, QImage.Format_RGB888
            )

        qimg = qimg.copy()
        pix = QPixmap.fromImage(qimg)

        # Scale to fit the available space inside the label while keeping aspect ratio.
        target_size = self.image_label.contentsRect().size()
        if target_size.width() > 0 and target_size.height() > 0:
            pix = pix.scaled(
                target_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )

        self.image_label.setPixmap(pix)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.camera_id)
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Per-camera context (real or simulated)
# ---------------------------------------------------------------------------

@dataclass
class CameraContext:
    camera_id: str
    name: str
    backend: Any
    thread: GrabThread
    tile: CameraTile
    hardware_id: Optional[str] = None  # serial / id / etc., used for persistence
    latest_frame: Optional[np.ndarray] = None
    recording: bool = False
    is_simulated: bool = False

    # Recording state for simulated cameras (and optionally for bookkeeping)
    session_dir: Optional[Path] = None
    session_id: Optional[str] = None
    video_writer: Optional[Any] = None
    frame_index: int = 0
    snapshot_fps: float = 0.0
    last_snapshot_time: float = 0.0
    record_video: bool = True
    video_fps: float = 30.0
    video_codec: str = "XVID"

    # NEW: PTV state
    ptv_tracker: Optional[PTVTracker] = None
    ptv_enabled: bool = False



# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MultiCamWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Multi-Camera Harvester GUI")
        self.resize(1400, 800)
        self.setStyleSheet("background-color: #f0f0f0;")

        self.cameras: Dict[str, CameraContext] = {}
        self.active_camera_id: Optional[str] = None

        # Global recording/session defaults
        self.save_root: Path = Path("captures").absolute()
        self.session_id_base: str = time.strftime("%Y%m%d_%H%M%S")

        # Persistent camera names (hardware_id -> label)
        self.camera_name_map: Dict[str, str] = self._load_camera_names()

        # Settings widgets
        self.active_cam_label: Optional[QLabel] = None
        self.hardware_id_label: Optional[QLabel] = None
        self.camera_name_edit: Optional[QLineEdit] = None
        self.camera_name_apply_btn: Optional[QPushButton] = None

        self.save_dir_edit: Optional[QLineEdit] = None
        self.session_id_edit: Optional[QLineEdit] = None
        self.record_check: Optional[QCheckBox] = None
        self.snapshot_fps_spin: Optional[QDoubleSpinBox] = None
        self.capture_fps_spin: Optional[QDoubleSpinBox] = None
        self.codec_combo: Optional[QComboBox] = None

        self.gain_edit: Optional[QLineEdit] = None
        self.gain_apply_btn: Optional[QPushButton] = None
        self.exposure_edit: Optional[QLineEdit] = None
        self.exposure_apply_btn: Optional[QPushButton] = None
        self.preview_fps_spin: Optional[QDoubleSpinBox] = None

        self._build_central_widget()
        self._build_docks()
        self._build_toolbar()
        self._build_status_bar()

        # Initialize real cams (if any) and simulated ones
        self._init_real_cameras()
        if ENABLE_SIMULATED_CAMERAS and NUM_SIMULATED_CAMERAS > 0:
            self._init_simulated_cameras(NUM_SIMULATED_CAMERAS)

        self._layout_tiles()

        # Select first camera by default (no auto-connect)
        if self.cameras:
            first_id = next(iter(self.cameras.keys()))
            self._set_active_camera(first_id)

    def _safe_dev_attr(self, dev_info, attr: str):
        """
        Safely read a GenTL/Harvester DeviceInfo attribute.

        Some properties (e.g. user_defined_name) may raise NotImplementedException
        instead of simply being absent. This wraps getattr() in try/except
        so we never crash during enumeration.
        """
        try:
            return getattr(dev_info, attr)
        except Exception:
            return None

    # ---- UI building --------------------------------------------------------

    def _build_central_widget(self):
        central = QWidget()
        self.grid_layout = QGridLayout(central)
        self.grid_layout.setContentsMargins(8, 8, 8, 8)
        self.grid_layout.setSpacing(8)
        self.setCentralWidget(central)

    def _build_docks(self):
        # Settings dock
        self.settings_dock = QDockWidget("Settings", self)
        self.settings_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.settings_dock.setAttribute(Qt.WA_DeleteOnClose, False)

        settings_widget = QWidget()
        s_layout = QVBoxLayout(settings_widget)
        s_layout.setContentsMargins(8, 8, 8, 8)
        s_layout.setSpacing(8)

        # Active camera info
        self.active_cam_label = QLabel("Active camera: (none)")
        self.active_cam_label.setStyleSheet("color: #333333; font-weight: bold;")
        s_layout.addWidget(self.active_cam_label)

        self.hardware_id_label = QLabel("Hardware ID: (none)")
        self.hardware_id_label.setStyleSheet("color: #555555;")
        s_layout.addWidget(self.hardware_id_label)

        # Camera name (persistent)
        name_form = QFormLayout()
        name_row = QHBoxLayout()
        self.camera_name_edit = QLineEdit()
        self.camera_name_apply_btn = QPushButton("Rename")
        self.camera_name_apply_btn.clicked.connect(self._apply_camera_name)
        name_row.addWidget(self.camera_name_edit)
        name_row.addWidget(self.camera_name_apply_btn)
        name_form.addRow("Camera name:", name_row)
        s_layout.addLayout(name_form)

        # Global recording/session settings
        rec_box = QGroupBox("Recording session (all cameras)")
        rec_form = QFormLayout()

        # Save directory
        self.save_dir_edit = QLineEdit(str(self.save_root))
        browse_btn = QPushButton("...")
        browse_btn.setMaximumWidth(30)
        browse_btn.clicked.connect(self._choose_save_dir)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.save_dir_edit)
        dir_row.addWidget(browse_btn)
        rec_form.addRow("Save dir:", dir_row)

        # Session ID
        self.session_id_edit = QLineEdit(self.session_id_base)
        rec_form.addRow("Session ID:", self.session_id_edit)

        # Record video
        self.record_check = QCheckBox("Record video")
        self.record_check.setChecked(True)
        rec_form.addRow("Video:", self.record_check)

        # Snapshot FPS (per second)
        self.snapshot_fps_spin = QDoubleSpinBox()
        self.snapshot_fps_spin.setRange(0.0, 120.0)
        self.snapshot_fps_spin.setDecimals(1)
        self.snapshot_fps_spin.setSingleStep(0.5)
        self.snapshot_fps_spin.setValue(0.0)
        self.snapshot_fps_spin.setToolTip(
            "Snapshots per second per camera while recording (0 = disabled)."
        )
        rec_form.addRow("Snapshot FPS:", self.snapshot_fps_spin)

        # Recording FPS (video_fps)
        self.capture_fps_spin = QDoubleSpinBox()
        self.capture_fps_spin.setRange(0.1, 120.0)
        self.capture_fps_spin.setDecimals(1)
        self.capture_fps_spin.setSingleStep(1.0)
        self.capture_fps_spin.setValue(30.0)
        self.capture_fps_spin.setToolTip(
            "Frame rate written into video files (frames per second)."
        )
        rec_form.addRow("Recording FPS:", self.capture_fps_spin)

        # Codec
        self.codec_combo = QComboBox()
        self.codec_combo.addItem("XVID (AVI)", "XVID")
        self.codec_combo.addItem("MJPG (AVI)", "MJPG")
        self.codec_combo.addItem("MP4V (MP4)", "MP4V")
        self.codec_combo.setCurrentIndex(0)
        rec_form.addRow("Video codec:", self.codec_combo)

        rec_box.setLayout(rec_form)
        s_layout.addWidget(rec_box)

        # Per-camera controls
        cam_box = QGroupBox("Active camera controls")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)

        # Gain
        self.gain_edit = QLineEdit()
        self.gain_apply_btn = QPushButton("Apply")
        self.gain_apply_btn.clicked.connect(self._apply_gain)
        gain_row = QHBoxLayout()
        gain_row.addWidget(self.gain_edit)
        gain_row.addWidget(self.gain_apply_btn)
        form.addRow("Gain:", gain_row)

        # Exposure
        self.exposure_edit = QLineEdit()
        self.exposure_apply_btn = QPushButton("Apply")
        self.exposure_apply_btn.clicked.connect(self._apply_exposure)
        exp_row = QHBoxLayout()
        exp_row.addWidget(self.exposure_edit)
        exp_row.addWidget(self.exposure_apply_btn)
        form.addRow("Exposure:", exp_row)

        # Preview FPS (works for real + sim)
        self.preview_fps_spin = QDoubleSpinBox()
        self.preview_fps_spin.setRange(0.5, 60.0)
        self.preview_fps_spin.setDecimals(1)
        self.preview_fps_spin.setSingleStep(0.5)
        self.preview_fps_spin.setValue(PREVIEW_FPS_REAL)
        self.preview_fps_spin.setToolTip(
            "How often the preview image is updated (frames per second)."
        )
        self.preview_fps_spin.valueChanged.connect(self._on_preview_fps_changed)
        form.addRow("Preview FPS:", self.preview_fps_spin)

        cam_box.setLayout(form)
        s_layout.addWidget(cam_box)

        s_layout.addStretch(1)

        self.settings_dock.setWidget(settings_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.settings_dock)
        self.settings_dock.hide()

        # Initially, gain/exposure disabled (no active real camera yet)
        self._set_gain_exposure_enabled(False)
        self._set_name_widgets_enabled(False)

        # Calibration dock using your existing CalibrationPanel
        self.calib_dock = QDockWidget("Calibration", self)
        self.calib_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.calib_dock.setAttribute(Qt.WA_DeleteOnClose, False)

        self.calib_panel = CalibrationPanel(self, self._get_active_frame)
        self.calib_dock.setWidget(self.calib_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.calib_dock)
        self.calib_dock.hide()

    def _build_toolbar(self):
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setStyleSheet("background: #f5f5f5;")
        self.addToolBar(toolbar)

        # Global connect/disconnect toggle
        self.connect_toggle_action = QAction("Connect all", self)
        self.connect_toggle_action.setStatusTip("Open all real cameras and start sim streams")
        self.connect_toggle_action.setCheckable(True)
        self.connect_toggle_action.setChecked(False)
        self.connect_toggle_action.triggered.connect(self._on_toggle_connect_all)
        toolbar.addAction(self.connect_toggle_action)

        toolbar.addSeparator()

        # Global recording toggle
        self.record_toggle_action = QAction("Start all recording", self)
        self.record_toggle_action.setStatusTip("Start recording on all cameras")
        self.record_toggle_action.setCheckable(True)
        self.record_toggle_action.setChecked(False)
        self.record_toggle_action.triggered.connect(self._on_toggle_record_all)
        toolbar.addAction(self.record_toggle_action)

        toolbar.addSeparator()

                # NEW: Global PTV toggle
        self.ptv_toggle_action = QAction("Start PTV", self)
        self.ptv_toggle_action.setCheckable(True)
        self.ptv_toggle_action.setChecked(False)
        self.ptv_toggle_action.setStatusTip("Start PTV tracking on all cameras")
        self.ptv_toggle_action.triggered.connect(self._on_toggle_ptv_all)
        toolbar.addAction(self.ptv_toggle_action)

        toolbar.addSeparator()


        self.snapshot_action = QAction("Snapshot (active)", self)
        self.snapshot_action.setStatusTip("Save a single snapshot from the active camera")
        self.snapshot_action.triggered.connect(self._on_snapshot_active)
        toolbar.addAction(self.snapshot_action)

        toolbar.addSeparator()
        # Settings dock toggle
        self.settings_action = QAction("Settings", self)
        self.settings_action.setCheckable(True)
        self.settings_action.setChecked(False)
        self.settings_action.setStatusTip("Show/hide settings panel")
        self.settings_action.toggled.connect(self.settings_dock.setVisible)
        toolbar.addAction(self.settings_action)


        self.calib_action = QAction("Calibration", self)
        self.calib_action.setCheckable(True)
        self.calib_action.setChecked(False)
        self.calib_action.setStatusTip("Show/hide calibration panel")
        self.calib_action.toggled.connect(self.calib_dock.setVisible)
        toolbar.addAction(self.calib_action)
        self.calib_dock.visibilityChanged.connect(self.calib_action.setChecked)

    def _build_status_bar(self):
        self.status = QStatusBar()
        self.status.setStyleSheet("background: #ffffff; color: #333333;")
        self.setStatusBar(self.status)
        self.status.showMessage("Ready (no cameras connected).")

    # ---- Camera initialization ----------------------------------------------

    def _init_real_cameras(self):
        """Enumerate real cameras via Harvester and create contexts."""
        if not ENABLE_REAL_CAMERAS:
            return

        if Harvester is None:
            self.status.showMessage("Harvester not installed; skipping real cameras.")
            return

        h = Harvester()
        try:
            h.add_file(CTI_PATH)
            h.update()
        except Exception as e:
            self.status.showMessage(f"Failed to init Harvester with CTI '{CTI_PATH}': {e}")
            h.reset()
            return

        n = len(h.device_info_list)
        if n == 0:
            self.status.showMessage("No real cameras detected via GenTL/CTI.")
            h.reset()
            return

        for idx, dev_info in enumerate(h.device_info_list):
            cam_id = f"REAL-{idx+1:02d}"

            # Use safe accessor for everything that might touch GenTL
            serial = self._safe_dev_attr(dev_info, "serial_number") \
                     or self._safe_dev_attr(dev_info, "serial")
            unique_id = self._safe_dev_attr(dev_info, "id_") \
                        or self._safe_dev_attr(dev_info, "unique_id")
            user_name = self._safe_dev_attr(dev_info, "user_defined_name")
            disp_name = self._safe_dev_attr(dev_info, "display_name")

            hardware_id = serial or unique_id or user_name or disp_name or cam_id
            default_name = user_name or f"RealCam {idx+1}"
            name = self.camera_name_map.get(str(hardware_id), default_name)

            backend = HarvesterCameraManager(cti_path=CTI_PATH, device_index=idx)
            tile = CameraTile(cam_id, name)
            thread = GrabThread(cam_id, backend, preview_fps=PREVIEW_FPS_REAL, parent=self)
            thread.frame_ready.connect(self._on_frame_ready)

            ctx = CameraContext(
                camera_id=cam_id,
                name=name,
                backend=backend,
                thread=thread,
                tile=tile,
                hardware_id=str(hardware_id),
                is_simulated=False,
            )

            ctx.ptv_tracker = None
            ctx.ptv_enabled = False

            self.cameras[cam_id] = ctx
            tile.clicked.connect(self._on_tile_clicked)


        h.reset()
        self.status.showMessage(f"Detected {n} real camera(s).")


    def _init_simulated_cameras(self, num: int):
        """Create N simulated cameras, tiles, and threads."""
        for idx in range(num):
            cam_id = f"SIM-{idx+1:02d}"
            default_name = f"SimCam {idx+1}"
            hardware_id = cam_id  # stable key for persistence
            name = self.camera_name_map.get(hardware_id, default_name)

            backend = SimulatedCameraBackend(cam_id, name, fps=PREVIEW_FPS_SIM)
            tile = CameraTile(cam_id, name)
            thread = GrabThread(cam_id, backend, preview_fps=PREVIEW_FPS_SIM, parent=self)
            thread.frame_ready.connect(self._on_frame_ready)

            ctx = CameraContext(
                camera_id=cam_id,
                name=name,
                backend=backend,
                thread=thread,
                tile=tile,
                hardware_id=hardware_id,
                is_simulated=True,
            )
            self.cameras[cam_id] = ctx
            tile.clicked.connect(self._on_tile_clicked)

        self.status.showMessage(
            f"Initialized {num} simulated camera(s) (plus any real cameras)."
        )

    def _layout_tiles(self):
        """Place camera tiles in a grid chosen to match target aspect ratio."""
        # Clear any existing items
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        n = len(self.cameras)
        if n <= 0:
            return

        # Get central widget size; fall back if not yet laid out
        cw = max(1, self.centralWidget().width())
        ch = max(1, self.centralWidget().height())

        best_rows, best_cols = 1, n
        best_err = float("inf")

        # Try all possible row counts, compute corresponding cols, pick the best match
        for rows in range(1, n + 1):
            cols = (n + rows - 1) // rows  # ceil(n / rows)
            cell_w = cw / max(1, cols)
            cell_h = ch / max(1, rows)
            cell_ar = cell_w / cell_h
            err = abs((cell_ar - TARGET_TILE_ASPECT) / TARGET_TILE_ASPECT)
            if err < best_err:
                best_err = err
                best_rows, best_cols = rows, cols

        rows, cols = best_rows, best_cols

        # Apply tiles
        ids = list(self.cameras.keys())
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= n:
                    break
                cam_id = ids[idx]
                tile = self.cameras[cam_id].tile
                self.grid_layout.addWidget(tile, r, c)
                idx += 1

        # Make rows and columns share extra space evenly
        for r in range(rows):
            self.grid_layout.setRowStretch(r, 1)
        for c in range(cols):
            self.grid_layout.setColumnStretch(c, 1)

    # ---- Helpers ------------------------------------------------------------

    def _load_camera_names(self) -> Dict[str, str]:
        """Load persistent camera name mapping from disk."""
        try:
            if CONFIG_PATH.exists():
                data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                names = data.get("camera_names", {})
                if isinstance(names, dict):
                    return {str(k): str(v) for k, v in names.items()}
        except Exception:
            pass
        return {}

    def _save_camera_names(self):
        """Save persistent camera name mapping to disk."""
        data = {"camera_names": self.camera_name_map}
        try:
            CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _set_name_widgets_enabled(self, enabled: bool):
        if self.camera_name_edit is not None:
            self.camera_name_edit.setEnabled(enabled)
        if self.camera_name_apply_btn is not None:
            self.camera_name_apply_btn.setEnabled(enabled)

    def _choose_save_dir(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Recording location",
            str(self.save_root),
        )
        if path:
            self.save_root = Path(path)
            if self.save_dir_edit is not None:
                self.save_dir_edit.setText(str(self.save_root))
            self.status.showMessage(f"Save dir: {self.save_root}")

    def _session_params(self) -> (Path, str):
        """Return (root_dir, session_id_base) from UI with sensible defaults."""
        root_text = self.save_dir_edit.text().strip() if self.save_dir_edit else ""
        root = Path(root_text) if root_text else self.save_root
        base_text = self.session_id_edit.text().strip() if self.session_id_edit else ""
        base = base_text or time.strftime("%Y%m%d_%H%M%S")
        return root, base

    def _get_active_frame(self):
        """For CalibrationPanel: return latest frame of active camera."""
        if self.active_camera_id and self.active_camera_id in self.cameras:
            return self.cameras[self.active_camera_id].latest_frame
        return None

    def _current_context(self) -> Optional[CameraContext]:
        if self.active_camera_id and self.active_camera_id in self.cameras:
            return self.cameras[self.active_camera_id]
        return None

    def _set_gain_exposure_enabled(self, enabled: bool):
        for w in (self.gain_edit, self.gain_apply_btn, self.exposure_edit, self.exposure_apply_btn):
            if w is not None:
                w.setEnabled(enabled)

    def _refresh_camera_params(self, silent: bool = False):
        """Pull gain/exposure from backend for the active camera, if real."""
        ctx = self._current_context()
        if ctx is None:
            self._set_gain_exposure_enabled(False)
            if self.gain_edit:
                self.gain_edit.setText("")
            if self.exposure_edit:
                self.exposure_edit.setText("")
            return

        # Preview FPS: always valid
        if self.preview_fps_spin is not None:
            self.preview_fps_spin.blockSignals(True)
            self.preview_fps_spin.setValue(getattr(ctx.thread, "_preview_fps", PREVIEW_FPS_REAL))
            self.preview_fps_spin.blockSignals(False)

        if ctx.is_simulated:
            # Simulated: no real gain/exposure
            self._set_gain_exposure_enabled(False)
            if self.gain_edit:
                self.gain_edit.setText("N/A")
            if self.exposure_edit:
                self.exposure_edit.setText("N/A")
            return

        # Real camera: try to query gain/exposure
        if ctx.backend is None:
            self._set_gain_exposure_enabled(False)
            return

        try:
            g = ctx.backend.get_gain()
            ex = ctx.backend.get_exposure()
            if self.gain_edit:
                self.gain_edit.setText(f"{g:.2f}")
            if self.exposure_edit:
                self.exposure_edit.setText(f"{ex:.1f}")
            self._set_gain_exposure_enabled(True)
        except Exception as e:
            if not silent:
                self.status.showMessage(f"Failed to refresh camera params: {e}")
            self._set_gain_exposure_enabled(False)

    # ---- Slots / event handlers ---------------------------------------------

    def _on_tile_clicked(self, camera_id: str):
        if camera_id not in self.cameras:
            return
        self._set_active_camera(camera_id)

    def _set_active_camera(self, camera_id: str):
        self.active_camera_id = camera_id
        for cid, ctx in self.cameras.items():
            ctx.tile.set_active(cid == camera_id)

        ctx = self.cameras[camera_id]
        self.status.showMessage(f"Active camera: {ctx.name} ({ctx.camera_id})")
        if self.active_cam_label:
            self.active_cam_label.setText(f"Active camera: {ctx.name} ({ctx.camera_id})")
        if self.hardware_id_label:
            hw = ctx.hardware_id or "N/A"
            self.hardware_id_label.setText(f"Hardware ID: {hw}")
        if self.camera_name_edit:
            self.camera_name_edit.setText(ctx.name)
        self._set_name_widgets_enabled(True)

        # Refresh settings for the newly selected camera
        self._refresh_camera_params(silent=True)

    def _on_frame_ready(self, camera_id: str, frame: np.ndarray):
        ctx = self.cameras.get(camera_id)
        if ctx is None:
            return

        # Store the latest frame for snapshots/calibration (RAW)
        ctx.latest_frame = frame

        # Apply PTV overlay if we have a tracker
        preview_frame = frame
        ptv = getattr(ctx, "ptv_tracker", None)
        if ptv is not None:
            # process_frame(copy=True) keeps ctx.latest_frame raw
            preview_frame = ptv.process_frame(frame, copy=True)

        # Update the tile preview
        ctx.tile.update_frame(preview_frame)



    def _on_preview_fps_changed(self, value: float):
        ctx = self._current_context()
        if ctx is None:
            return
        ctx.thread.set_preview_fps(value)

    def _on_toggle_ptv_all(self, checked: bool):
        """
        Global PTV toggle.

        checked = True  -> Start PTV: clear old traces and begin new session.
        checked = False -> Stop PTV: stop updating, but keep traces visible.
        """
        for ctx in self.cameras.values():
            # Lazily create a tracker if we don't have one
            if ctx.ptv_tracker is None and checked:
                ctx.ptv_tracker = PTVTracker(PTVConfig())

            ptv = ctx.ptv_tracker
            if ptv is None:
                continue

            if checked:
                # Start PTV: clear existing traces and begin tracking anew
                ptv.start()
                ctx.ptv_enabled = True
            else:
                # Stop PTV: stop updating, but traces remain and continue to be drawn
                ptv.stop()
                ctx.ptv_enabled = False

        # Update button text / tooltip
        if checked:
            self.ptv_toggle_action.setText("Stop PTV")
            self.ptv_toggle_action.setStatusTip(
                "Stop PTV (traces stay visible on frames)"
            )
        else:
            self.ptv_toggle_action.setText("Start PTV")
            self.ptv_toggle_action.setStatusTip(
                "Start new PTV session (clears old traces and starts tracking)"
            )


    # ---- Settings actions ---------------------------------------------------

    def _apply_gain(self):
        ctx = self._current_context()
        if ctx is None or ctx.is_simulated or ctx.backend is None:
            return
        if not self.gain_edit:
            return
        text = self.gain_edit.text().strip()
        if not text or text.upper() == "N/A":
            return
        try:
            val = float(text)
        except ValueError:
            self.status.showMessage("Gain must be a number.")
            return
        try:
            ctx.backend.set_gain(val)
            self.status.showMessage(f"{ctx.name}: Gain set to {val}")
        except Exception as e:
            self.status.showMessage(f"Failed to set gain: {e}")

    def _apply_exposure(self):
        ctx = self._current_context()
        if ctx is None or ctx.is_simulated or ctx.backend is None:
            return
        if not self.exposure_edit:
            return
        text = self.exposure_edit.text().strip()
        if not text or text.upper() == "N/A":
            return
        try:
            val = float(text)
        except ValueError:
            self.status.showMessage("Exposure must be a number.")
            return
        try:
            ctx.backend.set_exposure(val)
            self.status.showMessage(f"{ctx.name}: Exposure set to {val}")
        except Exception as e:
            self.status.showMessage(f"Failed to set exposure: {e}")

    def _apply_camera_name(self):
        ctx = self._current_context()
        if ctx is None or self.camera_name_edit is None:
            return
        new_name = self.camera_name_edit.text().strip()
        if not new_name:
            return

        ctx.name = new_name
        ctx.tile.name_label.setText(new_name)
        if self.active_cam_label:
            self.active_cam_label.setText(f"Active camera: {ctx.name} ({ctx.camera_id})")

        if ctx.hardware_id:
            self.camera_name_map[str(ctx.hardware_id)] = new_name
            self._save_camera_names()
            self.status.showMessage(f"Saved name '{new_name}' for {ctx.hardware_id}")
        else:
            self.status.showMessage(f"Renamed active camera to '{new_name}'")

    # ---- Global actions -----------------------------------------------------
    def _on_toggle_connect_all(self, checked: bool):
        """
        Toggle handler for the global connect/disconnect action.

        When checked=True, connect/stream all cameras.
        When checked=False, disconnect/stop all cameras.
        """
        if checked:
            # Going from disconnected -> connected
            self._on_connect_all()
            # Update label/status so it's obvious we're currently connected
            self.connect_toggle_action.setText("Disconnect all")
            self.connect_toggle_action.setStatusTip("Close all cameras / stop streams & recording")
        else:
            # Going from connected -> disconnected
            self._on_disconnect_all()
            self.connect_toggle_action.setText("Connect all")
            self.connect_toggle_action.setStatusTip("Open all real cameras and start sim streams")

            # Disconnecting stops any recording; ensure the record toggle reflects that
            if hasattr(self, "record_toggle_action"):
                self.record_toggle_action.blockSignals(True)
                self.record_toggle_action.setChecked(False)
                self.record_toggle_action.blockSignals(False)
                self.record_toggle_action.setText("Start all recording")
                self.record_toggle_action.setStatusTip("Start recording on all cameras")


    def _on_toggle_record_all(self, checked: bool):
        """
        Toggle handler for the global start/stop recording action.

        When checked=True, start recording on all cameras.
        When checked=False, stop recording on all cameras.
        """
        if checked:
            self._on_start_all()
            self.record_toggle_action.setText("Stop all recording")
            self.record_toggle_action.setStatusTip("Stop recording on all cameras")
        else:
            self._on_stop_all()
            self.record_toggle_action.setText("Start all recording")
            self.record_toggle_action.setStatusTip("Start recording on all cameras")

    def _on_connect_all(self):
        """Start streaming + grab threads for all cameras (real + simulated)."""
        for ctx in self.cameras.values():
            if ctx.is_simulated:
                if not ctx.backend.streaming:
                    ctx.backend.start_streaming()
            else:
                # Real camera via HarvesterCameraManager
                if not ctx.backend.streaming:
                    try:
                        ctx.backend.open()
                    except Exception as e:
                        ctx.tile.set_status(f"Error: {e}")
                        continue

            if not ctx.thread.isRunning():
                ctx.thread.start()

            if ctx.recording:
                ctx.tile.set_status("Streaming (REC)")
            else:
                ctx.tile.set_status("Streaming")

        self.status.showMessage("All cameras: streaming.")
        # Refresh settings for current active camera (now that things are open)
        self._refresh_camera_params(silent=True)

    def _on_disconnect_all(self):
        """Stop streaming, recording, and grab threads for all cameras."""
        for ctx in self.cameras.values():
            # Stop recording first
            if ctx.recording:
                if ctx.is_simulated:
                    if ctx.video_writer is not None:
                        try:
                            ctx.video_writer.release()
                        except Exception:
                            pass
                        ctx.video_writer = None
                else:
                    try:
                        ctx.backend.stop()
                    except Exception:
                        pass
                ctx.recording = False

            # Stop streaming
            if ctx.is_simulated:
                ctx.backend.stop_streaming()
            else:
                try:
                    ctx.backend.close()
                except Exception:
                    pass

            ctx.tile.set_status("Disconnected")

            if ctx.thread.isRunning():
                ctx.thread.stop()

        self.status.showMessage("All cameras: disconnected.")
        self._set_gain_exposure_enabled(False)

    def _on_start_all(self):
        """Start recording on all cameras using global session settings."""
        root_dir, base_id = self._session_params()
        record_video = self.record_check.isChecked() if self.record_check else True
        snapshot_fps = float(self.snapshot_fps_spin.value()) if self.snapshot_fps_spin else 0.0
        video_fps = float(self.capture_fps_spin.value()) if self.capture_fps_spin else 30.0
        codec = self.codec_combo.currentData() if self.codec_combo else "XVID"
        codec = codec or "XVID"

        try:
            root_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.status.showMessage(f"Failed to create root dir: {e}")
            return

        for ctx in self.cameras.values():
            cam_dir = root_dir / base_id / ctx.camera_id
            try:
                cam_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            session_id = f"{base_id}_{ctx.camera_id}"
            ctx.session_dir = cam_dir
            ctx.session_id = session_id
            ctx.snapshot_fps = snapshot_fps
            ctx.last_snapshot_time = 0.0
            ctx.record_video = record_video
            ctx.video_fps = video_fps
            ctx.video_codec = codec
            ctx.frame_index = 0

            if ctx.is_simulated:
                # Simulated recording handled in _on_frame_ready
                ctx.recording = True
                ctx.video_writer = None
                if getattr(ctx.backend, "streaming", False):
                    ctx.tile.set_status("Streaming (REC)")
                else:
                    ctx.tile.set_status("Idle (REC)")
            else:
                # Real cameras: let backend handle file writing
                try:
                    ctx.backend.add_file(
                        session_dir=cam_dir,
                        session_id=session_id,
                        record_video=record_video,
                        video_fps=video_fps,
                        video_codec=codec,
                        snapshot_fps=snapshot_fps,
                    )
                    ctx.backend.start()
                    ctx.recording = True
                    if ctx.backend.streaming:
                        ctx.tile.set_status("Streaming (REC)")
                    else:
                        ctx.tile.set_status("Idle (REC)")
                except Exception as e:
                    ctx.recording = False
                    ctx.tile.set_status(f"REC error: {e}")

        self.status.showMessage(f"Recording started for all cameras under '{root_dir / base_id}'.")

    def _on_stop_all(self):
        """Stop recording on all cameras (preview can continue)."""
        for ctx in self.cameras.values():
            if not ctx.recording:
                continue

            if ctx.is_simulated:
                if ctx.video_writer is not None:
                    try:
                        ctx.video_writer.release()
                    except Exception:
                        pass
                    ctx.video_writer = None
            else:
                try:
                    ctx.backend.stop()
                except Exception:
                    pass

            ctx.recording = False
            # Show streaming / disconnected based on streaming flag
            if getattr(ctx.backend, "streaming", False):
                ctx.tile.set_status("Streaming")
            else:
                ctx.tile.set_status("Disconnected")

        self.status.showMessage("All cameras: recording stopped.")

    def _on_snapshot_active(self):
        """Save a single snapshot from the active camera."""
        ctx = self._current_context()
        if ctx is None or ctx.latest_frame is None:
            self.status.showMessage("No frame available for active camera.")
            return

        root_dir, base_id = self._session_params()
        cam_dir = root_dir / base_id / ctx.camera_id
        try:
            cam_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        snap_path = cam_dir / f"{base_id}_{ctx.camera_id}_snapshot_{timestamp}.png"

        try:
            cv2.imwrite(str(snap_path), ctx.latest_frame)
            self.status.showMessage(f"Snapshot saved: {snap_path}")
        except Exception as e:
            self.status.showMessage(f"Failed to save snapshot: {e}")

    # ---- Events / Cleanup ---------------------------------------------------

    def resizeEvent(self, event):
        """Relayout tiles on window resize to keep aspect ratios pleasant."""
        super().resizeEvent(event)
        self._layout_tiles()

    def closeEvent(self, event):
        # Stop all threads and backends cleanly
        for ctx in self.cameras.values():
            # Stop recording
            if ctx.recording:
                if ctx.is_simulated:
                    if ctx.video_writer is not None:
                        try:
                            ctx.video_writer.release()
                        except Exception:
                            pass
                        ctx.video_writer = None
                else:
                    try:
                        ctx.backend.stop()
                    except Exception:
                        pass
                ctx.recording = False

            # Stop streaming
            try:
                if ctx.is_simulated:
                    ctx.backend.stop_streaming()
                else:
                    ctx.backend.close()
            except Exception:
                pass

            if ctx.thread.isRunning():
                ctx.thread.stop()

        event.accept()


# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    win = MultiCamWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
