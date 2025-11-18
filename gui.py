#!/usr/bin/env python3
"""
PyQt5 GUI for Harvester-based Mako camera with embedded preview.

- Uses backend.HarvesterCameraManager for streaming/recording.
- Embeds live preview in the window (QLabel).
- Toolbar actions:
    - Connect
    - Start Recording
    - Stop Recording
    - Snapshot Now
    - Change Save Location
- Side panel:
    - CTI path (default: VimbaGigETL.cti)
    - Session ID
    - Snapshot every N frames while recording
    - Gain
    - Exposure
    - Auto white balance
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QDoubleValidator
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QLineEdit,
    QSpinBox,
    QCheckBox,
    QGroupBox,
    QFormLayout,
    QMessageBox,
    QStatusBar,
    QToolBar,
    QAction,
    QDockWidget,
    QDoubleSpinBox,   # <- add
    QComboBox,        # <- add (for codec selector, used later)
)



from backend import HarvesterCameraManager
from calibration_panel import CalibrationPanel 


DEFAULT_CTI_PATH = "VimbaGigETL.cti"


# ---------------------------------------------------------------------------
# Worker thread: grabs frames from backend and sends them to GUI
# ---------------------------------------------------------------------------

class GrabThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    status_msg = pyqtSignal(str)

    def __init__(self, backend: HarvesterCameraManager, preview_fps: float = 2.0, parent=None):
        super().__init__(parent)
        self._backend = backend
        self._running = False
        self._preview_fps = float(preview_fps)

    def set_preview_fps(self, fps: float):
        """Update preview FPS from the GUI thread."""
        self._preview_fps = max(0.1, float(fps))  # avoid division by zero


    def start_grabbing(self):
        self._running = True
        self.start()

    def stop_grabbing(self):
        self._running = False
        self.wait()

    def run(self):
        self.status_msg.emit("Grab thread started.")

        last_emit = 0.0

        while self._running:
            if not self._backend.streaming:
                self.msleep(5)
                continue

            frame = self._backend.fetch_next(timeout=0.1)
            if frame is None:
                self.msleep(5)
                continue

            now = time.time()
            # Use configurable preview FPS
            target_dt = 1.0 / max(0.1, float(self._preview_fps))
            if now - last_emit >= target_dt:
                last_emit = now
                self.frame_ready.emit(frame)

        self.status_msg.emit("Grab thread stopped.")




# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Mako Harvester GUI")
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("background: #f0f0f0;")

        # Backend + thread
        self.backend: Optional[HarvesterCameraManager] = None
        self.grab_thread: Optional[GrabThread] = None
        self.latest_frame: Optional[np.ndarray] = None

        # State
        self.save_path = str(Path("captures").absolute())

        # Central preview area
        self._build_central_widget()

        # Dock with controls
        self._build_dock()

        # NEW: Calibration dock
        self._build_calibration_dock()

        # Toolbar
        self._build_toolbar()

        # Status bar
        self.status = QStatusBar()
        self.status.setStyleSheet("background: white;")
        self.setStatusBar(self.status)
        self._update_status("Ready.")


    # ---- UI building --------------------------------------------------------

    def _on_preview_fps_changed(self, value: float):
        if self.grab_thread is not None:
            self.grab_thread.set_preview_fps(float(value))


    def _build_calibration_dock(self):
        self.calib_dock = QDockWidget("Calibration", self)
        self.calib_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # pass a lambda that returns the latest frame
        self.calib_panel = CalibrationPanel(self, lambda: self.latest_frame)
        self.calib_dock.setWidget(self.calib_panel)

        self.addDockWidget(Qt.RightDockWidgetArea, self.calib_dock)

        # Hidden by default → only camera view at startup
        self.calib_dock.hide()

    def _set_camera_controls_enabled(self, enabled: bool):
        """Enable/disable all camera-control widgets in one place."""
        for w in (
            self.gain_edit,
            self.gain_apply_btn,
            self.exposure_edit,
            self.exposure_apply_btn,
            self.refresh_cam_btn,
        ):
            w.setEnabled(enabled)


    def _build_central_widget(self):
        central = QWidget()
        layout = QVBoxLayout(central)

        self.view_label = QLabel("No image")
        self.view_label.setAlignment(Qt.AlignCenter)
        self.view_label.setMinimumSize(640, 480)
        self.view_label.setStyleSheet("background: #202020; color: #cccccc;")
        layout.addWidget(self.view_label)

        self.setCentralWidget(central)

    def _build_dock(self):
        # Keep a reference so we can toggle from the toolbar
        self.settings_dock = QDockWidget("Settings", self)
        self.settings_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        # Make sure closing the dock just hides it, not deletes its children
        self.settings_dock.setAttribute(Qt.WA_DeleteOnClose, False)

        dock_widget = QWidget()
        form_layout = QVBoxLayout(dock_widget)

        # --- Session settings group -----------------------------------------
        session_box = QGroupBox("Session Settings")
        session_form = QFormLayout()

        # CTI path
        self.cti_edit = QLineEdit(DEFAULT_CTI_PATH)
        self.cti_edit.setPlaceholderText("Path to GenTL .cti (e.g. VimbaGigETL.cti)")
        browse_cti_btn = QPushButton("...")
        browse_cti_btn.setMaximumWidth(30)
        browse_cti_btn.clicked.connect(self._browse_cti)
        cti_row = QHBoxLayout()
        cti_row.addWidget(self.cti_edit)
        cti_row.addWidget(browse_cti_btn)
        session_form.addRow("CTI:", cti_row)

        # Save directory
        self.dir_edit = QLineEdit(self.save_path)
        browse_dir_btn = QPushButton("...")
        browse_dir_btn.setMaximumWidth(30)
        browse_dir_btn.clicked.connect(self._change_folder)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.dir_edit)
        dir_row.addWidget(browse_dir_btn)
        session_form.addRow("Save dir:", dir_row)

        # Session ID
        self.session_id_edit = QLineEdit(time.strftime("%Y%m%d_%H%M%S"))
        session_form.addRow("Session ID:", self.session_id_edit)

        # Record video
        self.record_check = QCheckBox("Record video")
        self.record_check.setChecked(True)
        session_form.addRow("Video:", self.record_check)

        # Snapshot every N frames while recording
        self.snapshot_spin = QSpinBox()
        self.snapshot_spin.setRange(0, 10000)
        self.snapshot_spin.setValue(0)
        self.snapshot_spin.setToolTip(
            "Save frames at set FPS during recording (0 = disabled)."
        )
        session_form.addRow("Snapshot at FPS:", self.snapshot_spin)

        session_box.setLayout(session_form)

        # --- Camera controls group ------------------------------------------
        cam_box = QGroupBox("Camera Controls")
        cam_form = QFormLayout()

        # Gain
        self.gain_edit = QLineEdit()
        self.gain_edit.setValidator(QDoubleValidator(bottom=0.0))
        self.gain_edit.setPlaceholderText("e.g. 0.0 - 18.0")
        self.gain_apply_btn = QPushButton("Apply")
        self.gain_apply_btn.setEnabled(False)
        self.gain_apply_btn.clicked.connect(self._apply_gain)
        gain_row = QHBoxLayout()
        gain_row.addWidget(self.gain_edit)
        gain_row.addWidget(self.gain_apply_btn)
        cam_form.addRow("Gain:", gain_row)

        # Exposure
        self.exposure_edit = QLineEdit()
        self.exposure_edit.setValidator(QDoubleValidator(bottom=0.0))
        self.exposure_edit.setPlaceholderText("Exposure (e.g. 20000)")
        self.exposure_apply_btn = QPushButton("Apply")
        self.exposure_apply_btn.setEnabled(False)
        self.exposure_apply_btn.clicked.connect(self._apply_exposure)
        exp_row = QHBoxLayout()
        exp_row.addWidget(self.exposure_edit)
        exp_row.addWidget(self.exposure_apply_btn)
        cam_form.addRow("Exposure:", exp_row)

        # Preview FPS (GUI update rate)
        self.preview_fps_spin = QDoubleSpinBox()
        self.preview_fps_spin.setRange(0.5, 60.0)
        self.preview_fps_spin.setDecimals(1)
        self.preview_fps_spin.setSingleStep(0.5)
        self.preview_fps_spin.setValue(2.0)  # default 2 FPS
        self.preview_fps_spin.setToolTip(
            "How often the preview image is updated (frames per second). "
            "Lower = less CPU/load, higher = smoother preview."
        )
        self.preview_fps_spin.valueChanged.connect(self._on_preview_fps_changed)
        cam_form.addRow("Preview FPS:", self.preview_fps_spin)

        # Recording / capture FPS (video_fps in backend)
        self.capture_fps_spin = QDoubleSpinBox()
        self.capture_fps_spin.setRange(0.1, 120.0)
        self.capture_fps_spin.setDecimals(1)
        self.capture_fps_spin.setSingleStep(1.0)
        self.capture_fps_spin.setValue(30.0)  # current default
        self.capture_fps_spin.setToolTip(
            "Frame rate written into the video file (frames per second)."
        )
        cam_form.addRow("Recording FPS:", self.capture_fps_spin)

        # Video codec selector
        self.codec_combo = QComboBox()
        self.codec_combo.addItem("XVID (AVI)", "XVID")
        self.codec_combo.addItem("MJPG (AVI)", "MJPG")
        self.codec_combo.addItem("MP4V (MP4)", "MP4V")
        self.codec_combo.setCurrentIndex(0)  # default to XVID (current behavior)
        self.codec_combo.setToolTip(
            "Approximate compression vs raw frames:\n"
            "• XVID: ~5–10× smaller, good quality, AVI\n"
            "• MJPG: ~2–5× smaller, very simple/compatible, AVI\n"
            "• MP4V: ~10–20× smaller, higher compression, MP4"
        )
        cam_form.addRow("Video codec:", self.codec_combo)


        # Refresh params button
        self.refresh_cam_btn = QPushButton("Refresh from camera")
        self.refresh_cam_btn.setEnabled(False)
        self.refresh_cam_btn.clicked.connect(
            lambda: self._refresh_camera_params(silent=False)
        )
        cam_form.addRow("", self.refresh_cam_btn)

        cam_box.setLayout(cam_form)

        # Add groups to the dock layout
        form_layout.addWidget(session_box)
        form_layout.addWidget(cam_box)
        form_layout.addStretch(1)

        # Attach the content widget to the dock and the dock to the window
        self.settings_dock.setWidget(dock_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.settings_dock)

        # Initially, nothing is controllable until we connect
        self._set_camera_controls_enabled(False)

        # Hidden by default → only camera view at startup
        self.settings_dock.hide()




    def _build_toolbar(self):
        toolbar = QToolBar("Camera Toolbar", self)
        toolbar.setStyleSheet("background: white;")
        self.addToolBar(toolbar)

        # Connect / Disconnect toggle (logic in _on_connect)
        self.connect_action = QAction("Connect", self)
        self.connect_action.setStatusTip("Connect to camera via Harvester")
        self.connect_action.triggered.connect(self._on_connect)
        toolbar.addAction(self.connect_action)
        toolbar.addSeparator()
        # Start recording
        self.start_action = QAction("Start Recording", self)
        self.start_action.setStatusTip("Start recording video/snapshots")
        self.start_action.setEnabled(False)
        self.start_action.triggered.connect(self._on_start)
        toolbar.addAction(self.start_action)
        toolbar.addSeparator()
        # Stop recording
        self.stop_action = QAction("Stop Recording", self)
        self.stop_action.setStatusTip("Stop recording")
        self.stop_action.setEnabled(False)
        self.stop_action.triggered.connect(self._on_stop)
        toolbar.addAction(self.stop_action)
        toolbar.addSeparator()
        # Snapshot now
        self.snapshot_action = QAction("Snapshot", self)
        self.snapshot_action.setStatusTip("Save a single snapshot")
        self.snapshot_action.setEnabled(False)
        self.snapshot_action.triggered.connect(self._on_snapshot_now)
        toolbar.addAction(self.snapshot_action)

        # Separator
        toolbar.addSeparator()

        # Change folder (same as button)
        self.change_folder_action = QAction("Change save location", self)
        self.change_folder_action.setStatusTip("Change output directory")
        self.change_folder_action.triggered.connect(self._change_folder)
        toolbar.addAction(self.change_folder_action)

        toolbar.addSeparator()

        # Settings dock toggle
        self.settings_action = QAction("Settings", self)
        self.settings_action.setCheckable(True)
        self.settings_action.setChecked(False)  # hidden at startup
        self.settings_action.setStatusTip("Show/hide settings panel")
        self.settings_action.toggled.connect(self.settings_dock.setVisible)
        toolbar.addAction(self.settings_action)
        
        toolbar.addSeparator()
        # Keep action state in sync if user closes dock via [x]
        self.settings_dock.visibilityChanged.connect(self.settings_action.setChecked)

        # Calibration dock toggle
        self.calib_action = QAction("Calibration", self)
        self.calib_action.setCheckable(True)
        self.calib_action.setChecked(False)  # hidden at startup
        self.calib_action.setStatusTip("Show/hide calibration panel")
        self.calib_action.toggled.connect(self.calib_dock.setVisible)
        toolbar.addAction(self.calib_action)

        # Keep action state in sync if user closes dock via [x]
        self.calib_dock.visibilityChanged.connect(self.calib_action.setChecked)


    # ---- Helpers ------------------------------------------------------------

    def _update_status(self, msg: str):
        self.status.showMessage(msg)

    def _browse_cti(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select GenTL .cti",
            "",
            "GenTL Producers (*.cti)",
        )
        if path:
            self.cti_edit.setText(path)

    def _change_folder(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Picture / Video Location",
            self.dir_edit.text().strip() or "",
        )
        if path:
            self.save_path = path
            self.dir_edit.setText(path)
            self._update_status(f"Save path: {path}")

    # ---- Connect / start / stop / snapshot ---------------------------------

    def _on_connect(self):
        # If already connected → act as Disconnect
        if self.backend is not None:
            # Stop grab thread
            if self.grab_thread is not None:
                self.grab_thread.stop_grabbing()
                self.grab_thread = None

            # Stop recording (if any) but keep it safe
            try:
                self.backend.stop()
            except Exception:
                pass

            try:
                self.backend.close()
            except Exception:
                pass

            self.backend = None
            self._update_status("Disconnected.")

            # Update UI
            self.connect_action.setText("Connect")
            self.connect_action.setStatusTip("Connect to camera via Harvester")
            self.start_action.setEnabled(False)
            self.stop_action.setEnabled(False)
            self.snapshot_action.setEnabled(False)
            self._set_camera_controls_enabled(False)
            return

        # Otherwise act as Connect
        cti_path = self.cti_edit.text().strip()
        if not cti_path:
            QMessageBox.warning(self, "CTI needed", "Please select a GenTL .cti file.")
            return

        try:
            self.backend = HarvesterCameraManager(cti_path=cti_path, device_index=0)
            self.backend.open()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open camera:\n{e}")
            self.backend = None
            return

        self._update_status("Connected. Streaming started.")
        self.connect_action.setText("Disconnect")
        self.connect_action.setStatusTip("Disconnect from camera")

        self.start_action.setEnabled(True)
        self.stop_action.setEnabled(False)
        self.snapshot_action.setEnabled(True)

        # Enable camera controls now that backend is ready
        self._set_camera_controls_enabled(True)

        # Initial camera values into the fields
        self._refresh_camera_params(silent=True)

        # Start grab thread
        # Start grab thread
        self.grab_thread = GrabThread(
            self.backend,
            preview_fps=float(self.preview_fps_spin.value()),
            parent=self,
        )
        self.grab_thread.frame_ready.connect(self._on_frame_ready)
        self.grab_thread.status_msg.connect(self._update_status)
        self.grab_thread.start_grabbing()




    def _on_start(self):
        if not self.backend:
            return

        session_dir = Path(self.dir_edit.text().strip() or self.save_path)
        session_id = self.session_id_edit.text().strip() or time.strftime("%Y%m%d_%H%M%S")

        # Get settings from the UI
        video_fps = float(self.capture_fps_spin.value())
        video_codec = self.codec_combo.currentData() or "XVID"

        try:
            self.backend.add_file(
                session_dir=session_dir,
                session_id=session_id,
                record_video=self.record_check.isChecked(),
                video_fps=video_fps,
                video_codec=video_codec,                       # NEW
                snapshot_fps=float(self.snapshot_spin.value()),
            )

            self.backend.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start recording:\n{e}")
            return

        self._update_status(f"Recording started: {session_dir} / {session_id}")
        self.start_action.setEnabled(False)
        self.stop_action.setEnabled(True)



    def _on_stop(self):
        if not self.backend:
            return

        self.backend.stop()
        self._update_status("Recording stopped (preview still live).")
        self.start_action.setEnabled(True)
        self.stop_action.setEnabled(False)

    def _on_snapshot_now(self):
        if self.latest_frame is None:
            QMessageBox.information(self, "No frame", "No frame available yet.")
            return

        session_dir = Path(self.dir_edit.text().strip() or self.save_path)
        session_id = self.session_id_edit.text().strip() or time.strftime("%Y%m%d_%H%M%S")
        session_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        snap_path = session_dir / f"{session_id}_snapshot_{timestamp}.png"

        try:
            cv2.imwrite(str(snap_path), self.latest_frame)
            self._update_status(f"Snapshot saved: {snap_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save snapshot:\n{e}")

    # ---- Frame display ------------------------------------------------------

    def _on_frame_ready(self, frame: np.ndarray):
        self.latest_frame = frame
        self._display_frame(frame)

    def _display_frame(self, frame: np.ndarray):
        # Mono camera ⇒ 2D array
        if frame.ndim == 2:
            # simple contrast stretch for visibility
            min_val, max_val = int(frame.min()), int(frame.max())
            if max_val > min_val:
                disp = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            else:
                disp = frame
            disp = np.ascontiguousarray(disp)
            h, w = disp.shape
            bytes_per_line = w
            qimg = QImage(disp.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:
            # if ever color: BGR → RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Important: copy so QImage doesn't point into Harvester buffer
        qimg = qimg.copy()
        pix = QPixmap.fromImage(qimg)

        self.view_label.setPixmap(
            pix.scaled(
                self.view_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    # ---- Camera param helpers ----------------------------------------------

    def _refresh_camera_params(self, silent: bool = False):
        if not self.backend:
            return
        try:
            # Gain
            try:
                g = self.backend.get_gain()
                self.gain_edit.setText(f"{g:.2f}")
            except Exception as e:
                if not silent:
                    print("Gain not available:", e)

            # Exposure
            try:
                ex = self.backend.get_exposure()
                self.exposure_edit.setText(f"{ex:.1f}")
            except Exception as e:
                if not silent:
                    print("Exposure not available:", e)

            # Auto white balance
            try:
                awb = self.backend.get_white_balance_auto()
                self.awb_check.blockSignals(True)
                self.awb_check.setChecked(awb)
                self.awb_check.blockSignals(False)
            except Exception as e:
                if not silent:
                    print("White balance auto not available:", e)
        except Exception as e:
            if not silent:
                QMessageBox.critical(self, "Error", f"Failed to refresh camera params:\n{e}")

    def _apply_gain(self):
        if not self.backend:
            return
        text = self.gain_edit.text().strip()
        if not text:
            return
        try:
            val = float(text)
        except ValueError:
            QMessageBox.warning(self, "Invalid value", "Gain must be a number.")
            return
        try:
            self.backend.set_gain(val)
            self._update_status(f"Gain set to {val}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to set gain:\n{e}")

    def _apply_exposure(self):
        if not self.backend:
            return
        text = self.exposure_edit.text().strip()
        if not text:
            return
        try:
            val = float(text)
        except ValueError:
            QMessageBox.warning(self, "Invalid value", "Exposure must be a number.")
            return
        try:
            self.backend.set_exposure(val)
            self._update_status(f"Exposure set to {val}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to set exposure:\n{e}")

    def _apply_awb(self, checked: bool):
        if not self.backend:
            return
        try:
            self.backend.set_white_balance_auto(checked)
            self._update_status(f"Auto white balance set to {checked}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to set auto white balance:\n{e}")

    # ---- Cleanup ------------------------------------------------------------

    def closeEvent(self, event):
        # stop thread
        if self.grab_thread is not None:
            self.grab_thread.stop_grabbing()
            self.grab_thread = None

        # stop backend
        if self.backend is not None:
            try:
                self.backend.stop()
            except Exception:
                pass
            self.backend.close()
            self.backend = None

        event.accept()


# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
