#!/usr/bin/env python3
"""
ptv_full.py - Offline PTV Maker GUI

Features:
- Load one or more videos and/or an image folder (snapshots)
- Optional undistortion using a calibration JSON (same format as cam_calib.json)
- Configure PTV parameters (threshold, min_area, etc.)
- Run PTV offline to:
    * Create a video with tracers drawn
    * Create a velocity field image (quiver + optional contour)
    * Create a velocity vs time plot (static)
    * Create an animated velocity vs time video with fixed axes
    * Create a rolling-average speed vs time plot (static, window set in GUI)
    * Create an animated rolling-average speed video with fixed axes

Requires:
    - ptv.py in the same directory (providing PTVConfig, PTVTracker)
    - OpenCV, numpy, matplotlib, PyQt5
"""

import sys
import math
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QListWidget,
    QCheckBox,
    QDoubleSpinBox,
    QSpinBox,
    QTextEdit,
    QMessageBox,
)

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt

from ptv import PTVConfig, PTVTracker


class PTVMakerWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PTV Maker")
        self.resize(900, 700)

        self._default_output_dir = Path("ptv_output").absolute()
        self._default_output_dir.mkdir(parents=True, exist_ok=True)

        # Inputs
        self.input_videos: List[Path] = []
        self.image_folder: Optional[Path] = None
        self.calib_path: Optional[Path] = None

        # Calibration state
        self.calib_K: Optional[np.ndarray] = None
        self.calib_dist: Optional[np.ndarray] = None
        self.calib_newK: Optional[np.ndarray] = None
        self.calib_size: Optional[Tuple[int, int]] = None  # (w, h)

        # Frame size (w, h) from first processed frame
        self.frame_size: Optional[Tuple[int, int]] = None

        # PTV configuration instance (used by the offline tracker)
        self.ptv_cfg = PTVConfig()

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ---------------- Input sources ----------------
        input_box = QGroupBox("Inputs")
        input_layout = QVBoxLayout()

        # Video list
        video_row = QHBoxLayout()
        self.video_list = QListWidget()
        self.video_list.setSelectionMode(self.video_list.ExtendedSelection)

        video_btn_col = QVBoxLayout()
        self.add_video_btn = QPushButton("Add video...")
        self.add_video_btn.clicked.connect(self._add_video)
        self.remove_video_btn = QPushButton("Remove selected")
        self.remove_video_btn.clicked.connect(self._remove_selected_videos)
        video_btn_col.addWidget(self.add_video_btn)
        video_btn_col.addWidget(self.remove_video_btn)
        video_btn_col.addStretch(1)

        video_row.addWidget(self.video_list, stretch=1)
        video_row.addLayout(video_btn_col)
        input_layout.addLayout(video_row)

        # Image folder
        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Optional: folder of snapshot images")
        folder_btn = QPushButton("Browse...")
        folder_btn.clicked.connect(self._choose_image_folder)
        folder_row.addWidget(QLabel("Image folder:"))
        folder_row.addWidget(self.folder_edit, stretch=1)
        folder_row.addWidget(folder_btn)
        input_layout.addLayout(folder_row)

        input_box.setLayout(input_layout)
        main_layout.addWidget(input_box)

        # ---------------- Calibration + PTV config ----------------
        calib_box = QGroupBox("Calibration & PTV configuration")
        calib_form = QFormLayout()

        # Calibration file
        calib_row = QHBoxLayout()
        self.calib_edit = QLineEdit()
        self.calib_edit.setPlaceholderText("Optional: cam_calib.json")
        calib_browse_btn = QPushButton("Browse...")
        calib_browse_btn.clicked.connect(self._choose_calib_file)
        calib_row.addWidget(self.calib_edit)
        calib_row.addWidget(calib_browse_btn)
        calib_form.addRow("Calibration JSON:", calib_row)

        self.undistort_check = QCheckBox("Apply undistortion")
        self.undistort_check.setChecked(True)
        calib_form.addRow("", self.undistort_check)

        # PTV config controls (initialized from ptv_cfg)
        # Threshold
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.setValue(int(self.ptv_cfg.threshold))
        calib_form.addRow("Threshold (0–255):", self.threshold_spin)

        # Min area
        self.min_area_spin = QDoubleSpinBox()
        self.min_area_spin.setRange(0.0, 1e9)
        self.min_area_spin.setDecimals(2)
        self.min_area_spin.setSingleStep(1.0)
        self.min_area_spin.setValue(float(self.ptv_cfg.min_area))
        calib_form.addRow("Min blob area (px):", self.min_area_spin)

        # Max tracks
        self.max_tracks_spin = QSpinBox()
        self.max_tracks_spin.setRange(1, 100000)
        self.max_tracks_spin.setValue(int(self.ptv_cfg.max_tracks))
        calib_form.addRow("Max tracks:", self.max_tracks_spin)

        # Max history
        self.max_history_spin = QSpinBox()
        self.max_history_spin.setRange(1, 1_000_000)
        self.max_history_spin.setValue(int(self.ptv_cfg.max_history))
        calib_form.addRow("Max history length:", self.max_history_spin)

        # Max dist
        self.max_dist_spin = QDoubleSpinBox()
        self.max_dist_spin.setRange(0.0, 1e6)
        self.max_dist_spin.setDecimals(2)
        self.max_dist_spin.setSingleStep(1.0)
        self.max_dist_spin.setValue(float(self.ptv_cfg.max_dist_px))
        calib_form.addRow("Max jump between frames (px):", self.max_dist_spin)

        # Max missed frames
        self.max_missed_spin = QSpinBox()
        self.max_missed_spin.setRange(0, 100000)
        self.max_missed_spin.setValue(int(self.ptv_cfg.max_missed_frames))
        calib_form.addRow("Max missed frames:", self.max_missed_spin)

        # Process every Nth frame
        self.process_every_spin = QSpinBox()
        self.process_every_spin.setRange(1, 1000)
        self.process_every_spin.setValue(int(self.ptv_cfg.process_every_n_frames))
        calib_form.addRow("Process every Nth frame:", self.process_every_spin)

        # Downscale factor
        self.downscale_spin = QDoubleSpinBox()
        self.downscale_spin.setRange(0.05, 4.0)
        self.downscale_spin.setDecimals(2)
        self.downscale_spin.setSingleStep(0.05)
        self.downscale_spin.setValue(float(self.ptv_cfg.downscale_factor))
        calib_form.addRow("Downscale factor:", self.downscale_spin)

        calib_box.setLayout(calib_form)
        main_layout.addWidget(calib_box)

        # ---------------- Output settings ----------------
        out_box = QGroupBox("Output")
        out_form = QFormLayout()

        # Output directory
        out_dir_row = QHBoxLayout()
        self.out_dir_edit = QLineEdit(str(self._default_output_dir))
        out_dir_btn = QPushButton("Browse...")
        out_dir_btn.clicked.connect(self._choose_output_dir)
        out_dir_row.addWidget(self.out_dir_edit)
        out_dir_row.addWidget(out_dir_btn)
        out_form.addRow("Output dir:", out_dir_row)

        # Output base name
        self.out_base_edit = QLineEdit("ptv_run")
        out_form.addRow("Base name:", self.out_base_edit)

        # Frame rate (used only when no video FPS, or image-folder-only run)
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.1, 240.0)
        self.fps_spin.setDecimals(2)
        self.fps_spin.setSingleStep(1.0)
        self.fps_spin.setValue(30.0)
        self.fps_spin.setToolTip(
            "Frame rate used when processing image folders only, or if "
            "video FPS metadata is unavailable."
        )
        out_form.addRow("Frame rate (Hz):", self.fps_spin)

        # Rolling average window (seconds)
        self.ra_window_spin = QDoubleSpinBox()
        self.ra_window_spin.setRange(0.0, 1e4)
        self.ra_window_spin.setDecimals(2)
        self.ra_window_spin.setSingleStep(0.5)
        self.ra_window_spin.setValue(1.0)
        self.ra_window_spin.setToolTip(
            "Rolling-average window in seconds for speed vs time.\n"
            "Set to 0 to disable rolling-average plot/video."
        )
        out_form.addRow("Rolling avg window (s):", self.ra_window_spin)

        out_box.setLayout(out_form)
        main_layout.addWidget(out_box)

        # ---------------- Run button + log ----------------
        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run PTV")
        self.run_btn.clicked.connect(self._on_run)
        run_row.addWidget(self.run_btn)
        run_row.addStretch(1)
        main_layout.addLayout(run_row)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        main_layout.addWidget(self.log_edit, stretch=1)

        self.setCentralWidget(central)

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def log(self, msg: str):
        self.log_edit.append(msg)
        cursor = self.log_edit.textCursor()
        cursor.movePosition(cursor.End)
        self.log_edit.setTextCursor(cursor)
        QApplication.processEvents()

    def _add_video(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select video files",
            "",
            "Video files (*.avi *.mp4 *.mov *.mkv);;All files (*.*)",
        )
        if not paths:
            return
        for p in paths:
            path = Path(p)
            if path not in self.input_videos:
                self.input_videos.append(path)
                self.video_list.addItem(str(path))
        self.log(f"Added {len(paths)} video(s).")

    def _remove_selected_videos(self):
        items = self.video_list.selectedItems()
        for it in items:
            row = self.video_list.row(it)
            path_str = self.video_list.item(row).text()
            try:
                self.input_videos.remove(Path(path_str))
            except ValueError:
                pass
            self.video_list.takeItem(row)
        if items:
            self.log("Removed selected videos.")

    def _choose_image_folder(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select image folder (snapshots)",
            "",
        )
        if path:
            self.image_folder = Path(path)
            self.folder_edit.setText(str(self.image_folder))
            self.log(f"Image folder: {self.image_folder}")

    def _choose_calib_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select calibration JSON",
            "",
            "JSON files (*.json);;All files (*.*)",
        )
        if path:
            self.calib_path = Path(path)
            self.calib_edit.setText(str(self.calib_path))
            self.log(f"Calibration file: {self.calib_path}")

    def _choose_output_dir(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select output directory",
            str(self._default_output_dir),
        )
        if path:
            self.out_dir_edit.setText(path)

    # ------------------------------------------------------------------
    # Core configuration helpers
    # ------------------------------------------------------------------
    def _update_ptv_config_from_ui(self):
        self.ptv_cfg.threshold = int(self.threshold_spin.value())
        self.ptv_cfg.min_area = float(self.min_area_spin.value())
        self.ptv_cfg.max_tracks = int(self.max_tracks_spin.value())
        self.ptv_cfg.max_history = int(self.max_history_spin.value())
        self.ptv_cfg.max_dist_px = float(self.max_dist_spin.value())
        self.ptv_cfg.max_missed_frames = int(self.max_missed_spin.value())
        self.ptv_cfg.process_every_n_frames = int(self.process_every_spin.value())
        self.ptv_cfg.downscale_factor = float(self.downscale_spin.value())

    def _load_calibration_if_needed(self) -> None:
        self.calib_K = None
        self.calib_dist = None
        self.calib_newK = None
        self.calib_size = None

        if not self.calib_path or not self.undistort_check.isChecked():
            self.log("No calibration / undistortion disabled.")
            return

        try:
            with open(self.calib_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.log(f"[WARN] Could not load calibration: {e}")
            return

        try:
            K = np.array(data["intrinsic_matrix"], dtype=np.float32)
            dist = np.array(data["distortion_coeffs"], dtype=np.float32).reshape(-1, 1)
            w0 = int(data["image_width"])
            h0 = int(data["image_height"])
        except Exception as e:
            self.log(f"[WARN] Invalid calibration JSON format: {e}")
            return

        img_size = (w0, h0)
        newK, _roi = cv2.getOptimalNewCameraMatrix(
            K, dist, img_size, alpha=1.0, newImgSize=img_size
        )
        self.calib_K = K
        self.calib_dist = dist
        self.calib_newK = newK
        self.calib_size = img_size
        self.log(f"[INFO] Loaded calibration: {w0} x {h0}")

    def _iter_image_folder_frames(self) -> Tuple[int, List[Path]]:
        if not self.image_folder:
            return 0, []
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        files = [
            p for p in sorted(self.image_folder.iterdir())
            if p.is_file() and p.suffix.lower() in exts
        ]
        return len(files), files

    def _determine_effective_fps(self, user_fps: float) -> float:
        """
        Decide what FPS to use for velocities and output video.

        - If there are videos, try to read FPS from them.
          Use the FPS of the first video with valid metadata.
          Warn if subsequent videos differ by >5%.
        - If no valid FPS found from videos, fall back to user_fps.
        - If there are no videos at all (image folder only), use user_fps.
        """
        if self.input_videos:
            fps_values: List[Tuple[float, Path]] = []
            for vpath in self.input_videos:
                cap = cv2.VideoCapture(str(vpath))
                if not cap.isOpened():
                    self.log(f"[WARN] Could not open video to read FPS: {vpath}")
                    continue
                fps_v = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps_v and fps_v > 0:
                    fps_values.append((fps_v, vpath))

            if fps_values:
                base_fps, base_path = fps_values[0]
                self.log(f"[INFO] Using FPS {base_fps:.3f} from video: {base_path}")
                # Check for large mismatches
                for v_fps, vpath in fps_values[1:]:
                    if abs(v_fps - base_fps) / base_fps > 0.05:
                        self.log(
                            f"[WARN] Video {vpath} FPS {v_fps:.3f} differs from "
                            f"{base_path} FPS {base_fps:.3f}; using {base_fps:.3f} globally."
                        )
                        break
                return base_fps
            else:
                self.log(
                    "[WARN] Could not read FPS from any videos; "
                    f"falling back to user FPS = {user_fps:.3f}."
                )
                return user_fps
        else:
            self.log(
                f"[INFO] No videos loaded; using user FPS = {user_fps:.3f} "
                "for image-folder run."
            )
            return user_fps

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------
    def _on_run(self):
        if not self.input_videos and not self.image_folder:
            QMessageBox.warning(
                self, "PTV Maker",
                "Please add at least one video or choose an image folder."
            )
            return

        out_dir = Path(self.out_dir_edit.text().strip() or ".").absolute()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_base = self.out_base_edit.text().strip() or "ptv_run"
        user_fps = float(self.fps_spin.value())
        if user_fps <= 0:
            QMessageBox.warning(self, "PTV Maker", "Frame rate must be > 0.")
            return

        # Decide effective FPS (from video if available, else user)
        fps = self._determine_effective_fps(user_fps)
        if fps <= 0:
            QMessageBox.warning(self, "PTV Maker", "Effective FPS is invalid (<= 0).")
            return

        # Update config from UI, load calibration
        self._update_ptv_config_from_ui()
        self._load_calibration_if_needed()

        self.frame_size = None
        self.run_btn.setEnabled(False)
        try:
            self._run_ptv_pipeline(out_dir, out_base, fps)
        finally:
            self.run_btn.setEnabled(True)

    def _run_ptv_pipeline(self, out_dir: Path, out_base: str, fps: float):
        self.log(f"Starting PTV pipeline with FPS = {fps:.3f} ...")

        tracker = PTVTracker(self.ptv_cfg)
        tracker.start()

        # History: track_id -> list of (t_abs, x, y, vx, vy)
        history: Dict[int, List[Tuple[float, float, float, float, float]]] = {}
        prev_pos: Dict[int, Tuple[int, float, float]] = {}

        frame_index = 0
        total_frames = 0

        # Count frames in videos
        for vpath in self.input_videos:
            cap = cv2.VideoCapture(str(vpath))
            if not cap.isOpened():
                self.log(f"[WARN] Could not open video: {vpath}")
                continue
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames += max(count, 0)
            cap.release()

        # Count frames in image folder
        folder_count, folder_files = self._iter_image_folder_frames()
        total_frames += folder_count

        if total_frames == 0:
            self.log("[ERROR] No frames to process.")
            return

        self.log(f"[INFO] Total frames to process: {total_frames}")

        out_video = None
        out_size = None  # (w, h)

        def init_video_writer(sample_frame: np.ndarray):
            nonlocal out_video, out_size
            h, w = sample_frame.shape[:2]
            out_size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_path = out_dir / f"{out_base}_tracers.mp4"
            out_video = cv2.VideoWriter(str(video_path), fourcc, fps, out_size)
            self.log(f"[INFO] Writing tracer video to: {video_path}")

        def ensure_frame_size(frame: np.ndarray):
            if self.frame_size is None:
                h, w = frame.shape[:2]
                self.frame_size = (w, h)

        # Undistort helper
        def undistort_if_needed(frame: np.ndarray) -> np.ndarray:
            if (
                self.calib_K is None
                or self.calib_dist is None
                or self.calib_newK is None
            ):
                return frame
            h, w = frame.shape[:2]
            if self.calib_size and (w, h) != self.calib_size:
                self.log("[WARN] Frame size != calibration size – undistortion may be off.")
            return cv2.undistort(frame, self.calib_K, self.calib_dist, None, self.calib_newK)

        # ------------------------------------------------------------------
        # Process videos
        # ------------------------------------------------------------------
        for vpath in self.input_videos:
            cap = cv2.VideoCapture(str(vpath))
            if not cap.isOpened():
                self.log(f"[WARN] Skipping unopened video: {vpath}")
                continue

            self.log(f"[INFO] Processing video: {vpath}")

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame = undistort_if_needed(frame)
                ensure_frame_size(frame)
                frame_index += 1
                if frame_index % 10 == 0:
                    self.log(f"Frame {frame_index}/{total_frames} ...")

                if out_video is None:
                    init_video_writer(frame)

                out_frame = tracker.process_frame(frame)
                self._accumulate_history(
                    tracker, history, prev_pos, frame_index, fps
                )
                out_video.write(out_frame)

            cap.release()

        # ------------------------------------------------------------------
        # Process image folder (snapshots)
        # ------------------------------------------------------------------
        if folder_count > 0:
            self.log(f"[INFO] Processing image folder: {self.image_folder}")
            for fpath in folder_files:
                img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
                if img is None:
                    self.log(f"[WARN] Could not read image: {fpath}")
                    continue

                img = undistort_if_needed(img)
                ensure_frame_size(img)
                frame_index += 1
                if frame_index % 10 == 0:
                    self.log(f"Frame {frame_index}/{total_frames} ...")

                if out_video is None:
                    init_video_writer(img)

                out_frame = tracker.process_frame(img)
                self._accumulate_history(
                    tracker, history, prev_pos, frame_index, fps
                )
                out_video.write(out_frame)

        if out_video is not None:
            out_video.release()
            self.log("[INFO] Finished writing tracer video.")

        # ------------------------------------------------------------------
        # Post-processing: velocity field + velocity-time plots
        # ------------------------------------------------------------------
        if history:
            self._make_velocity_field_plot(history, out_dir, out_base)
            self._make_velocity_time_plots(history, out_dir, out_base)
            self.log("[INFO] Generated velocity field and velocity-time plots.")
        else:
            self.log("[WARN] No track history collected (no particles detected?).")

        self.log("[DONE] PTV pipeline complete.")

    # ------------------------------------------------------------------
    # History accumulation
    # ------------------------------------------------------------------
    def _accumulate_history(
        self,
        tracker: PTVTracker,
        history: Dict[int, List[Tuple[float, float, float, float, float]]],
        prev_pos: Dict[int, Tuple[int, float, float]],
        frame_index: int,
        fps: float,
    ):
        """
        Collect per-track positions and velocities at the current frame.

        history: track_id -> list of (t_abs, x, y, vx, vy)
        prev_pos: track_id -> (prev_frame_index, x_prev, y_prev)
        """
        dt_frame = 1.0 / fps
        t = frame_index * dt_frame

        for tr in tracker.tracks:
            if not tr.points:
                continue
            x, y = tr.points[-1]  # last point in this track

            if tr.id in prev_pos:
                prev_idx, px, py = prev_pos[tr.id]
                df = frame_index - prev_idx
                if df <= 0:
                    vx = vy = 0.0
                else:
                    dt = df * dt_frame
                    vx = (x - px) / dt
                    vy = (y - py) / dt
            else:
                vx = vy = 0.0

            prev_pos[tr.id] = (frame_index, x, y)
            history.setdefault(tr.id, []).append((t, x, y, vx, vy))

    # ------------------------------------------------------------------
    # Velocity field (quiver + optional contour)
    # ------------------------------------------------------------------
    def _make_velocity_field_plot(
        self,
        history: Dict[int, List[Tuple[float, float, float, float, float]]],
        out_dir: Path,
        out_base: str,
    ):
        """
        Create a velocity field image (quiver + optional contour) using
        average position and average velocity per track.

        Plot extents and figure size are matched to the input video size
        (if known), so the plot has the same width/height and uses
        pixel coordinates.
        """
        xs = []
        ys = []
        us = []
        vs = []

        # Build one arrow per track (average position & velocity)
        for samples in history.values():
            if len(samples) < 2:
                continue

            xs_track = []
            ys_track = []
            us_track = []
            vs_track = []

            # Skip the first sample (often vx = vy = 0 by construction)
            for (t, x, y, vx, vy) in samples[1:]:
                xs_track.append(x)
                ys_track.append(y)
                us_track.append(vx)
                vs_track.append(vy)

            if not xs_track:
                continue

            xs.append(float(np.mean(xs_track)))
            ys.append(float(np.mean(ys_track)))
            us.append(float(np.mean(us_track)))
            vs.append(float(np.mean(vs_track)))

        if not xs:
            self.log("[WARN] No velocity samples for velocity field plot.")
            return

        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)
        us = np.array(us, dtype=float)
        vs = np.array(vs, dtype=float)
        mag = np.sqrt(us * us + vs * vs)

        # Use frame size if available; otherwise fall back to data extents
        if self.frame_size is not None:
            w, h = self.frame_size
            x_min, x_max = 0.0, float(w)
            y_min, y_max = 0.0, float(h)
        else:
            x_min, x_max = float(xs.min()), float(xs.max())
            y_min, y_max = float(ys.min()), float(ys.max())

        width = max(x_max - x_min, 1.0)
        height = max(y_max - y_min, 1.0)

        # Figure size in inches so pixels roughly match video size
        if self.frame_size is not None:
            dpi = 100
            fig_w = max(width / dpi, 2.0)
            fig_h = max(height / dpi, 2.0)
        else:
            dpi = 100
            fig_w, fig_h = 8.0, 6.0

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax.set_title("PTV Velocity Field (pixels / second)")

        # Optional filled contour of speed using unstructured triangular grid
        # Only attempt if we have at least 3 points
        if len(xs) >= 3:
            try:
                contour = ax.tricontourf(xs, ys, mag, levels=20, cmap="viridis")
                fig.colorbar(contour, ax=ax, label="Speed [px/s]")
            except Exception as e:
                self.log(f"[WARN] Contour plot failed, using quiver only: {e}")
        else:
            self.log("[INFO] Too few points for contour plot; drawing quiver only.")

        # Quiver on top (always)
        ax.quiver(
            xs, ys, us, vs,
            angles="xy", scale_units="xy", scale=1.0,
            width=0.003,
            color="white",
        )

        # Axes in pixel coordinates, y-axis downwards like images
        ax.set_xlim(x_min - 0.05 * width, x_max + 0.05 * width)
        ax.set_ylim(y_max + 0.05 * height, y_min - 0.05 * height)
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px] (image coords, top-down)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

        out_path = out_dir / f"{out_base}_velocity_field.png"
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=dpi)
        plt.close(fig)

        self.log(f"[INFO] Saved velocity field to: {out_path}")

    # ------------------------------------------------------------------
    # Velocity vs time (static + animated + rolling-average)
    # ------------------------------------------------------------------
    def _make_velocity_time_plots(
        self,
        history: Dict[int, List[Tuple[float, float, float, float, float]]],
        out_dir: Path,
        out_base: str,
    ):
        """
        Create:
          - Static speed vs time plot (PNG)
          - Animated speed vs time video (MP4) with fixed axes
          - Static rolling-average speed vs time plot (if window > 0)
          - Animated rolling-average speed vs time video (if window > 0)
        """
        track_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        # Build relative times and speeds per track
        for track_id, samples in history.items():
            if len(samples) < 2:
                continue

            t0 = samples[0][0]
            rel_ts = []
            speeds = []

            for (t, _x, _y, vx, vy) in samples[1:]:
                rel_ts.append(t - t0)
                speeds.append(math.sqrt(vx * vx + vy * vy))

            if not rel_ts:
                continue

            track_data[track_id] = (np.array(rel_ts, dtype=float),
                                    np.array(speeds, dtype=float))

        if not track_data:
            self.log("[WARN] No tracks available for velocity-time plots.")
            return

        all_t = np.concatenate([v[0] for v in track_data.values()])
        all_s = np.concatenate([v[1] for v in track_data.values()])

        max_t = float(all_t.max())
        max_s = float(all_s.max()) if all_s.size > 0 else 0.0
        if max_s <= 0.0:
            max_s = 1.0

        # Get rolling-average window (seconds) from UI
        window_sec = 0.0
        if hasattr(self, "ra_window_spin") and self.ra_window_spin is not None:
            window_sec = float(self.ra_window_spin.value())
            if window_sec < 0.0:
                window_sec = 0.0

        # -------- Static plot: all individual speeds --------
        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
        ax.set_title("Tracer Speed vs Time")
        ax.set_xlabel("Time since track start [s]")
        ax.set_ylabel("Speed [px/s]")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)

        for t_rel, speeds in track_data.values():
            ax.plot(t_rel, speeds, alpha=0.6, linewidth=1.0)

        ax.set_xlim(0.0, max_t * 1.05)
        ax.set_ylim(0.0, max_s * 1.05)

        out_static = out_dir / f"{out_base}_velocity_vs_time.png"
        fig.tight_layout()
        fig.savefig(str(out_static), dpi=120)
        plt.close(fig)
        self.log(f"[INFO] Saved static velocity vs time plot to: {out_static}")

        # -------- Animated plot: all individual speeds --------
        fps_anim = 30.0
        num_frames = int(max(60, min(600, max_t * fps_anim))) if max_t > 0 else 60
        tau_values = np.linspace(0.0, max_t, num_frames)

        dpi_anim = 120
        fig_anim, ax_anim = plt.subplots(figsize=(8, 6), dpi=dpi_anim)
        ax_anim.set_title("Tracer Speed vs Time (animated)")
        ax_anim.set_xlabel("Time since track start [s]")
        ax_anim.set_ylabel("Speed [px/s]")
        ax_anim.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
        ax_anim.set_xlim(0.0, max_t * 1.05)
        ax_anim.set_ylim(0.0, max_s * 1.05)
        fig_anim.tight_layout()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        anim_path = out_dir / f"{out_base}_velocity_vs_time_anim.mp4"
        writer = None  # lazy-init once we know the canvas size

        self.log(f"[INFO] Writing animated velocity vs time to: {anim_path}")

        for i, tau in enumerate(tau_values):
            ax_anim.cla()
            ax_anim.set_title("Tracer Speed vs Time (animated)")
            ax_anim.set_xlabel("Time since track start [s]")
            ax_anim.set_ylabel("Speed [px/s]")
            ax_anim.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
            ax_anim.set_xlim(0.0, max_t * 1.05)
            ax_anim.set_ylim(0.0, max_s * 1.05)

            for t_rel, speeds in track_data.values():
                mask = t_rel <= tau
                if not np.any(mask):
                    continue
                ax_anim.plot(t_rel[mask], speeds[mask], alpha=0.6, linewidth=1.0)

            # Vertical time marker
            ax_anim.axvline(tau, color="k", linestyle="--", alpha=0.3)

            fig_anim.canvas.draw()

            # RGBA buffer from the canvas, convert to BGR for OpenCV
            buf = np.asarray(fig_anim.canvas.buffer_rgba())  # H x W x 4
            height_px, width_px, _ = buf.shape
            img_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)

            # Initialize writer once we know the actual size
            if writer is None:
                writer = cv2.VideoWriter(
                    str(anim_path), fourcc, fps_anim, (width_px, height_px)
                )

            writer.write(img_bgr)

            if (i + 1) % 20 == 0:
                self.log(f"Animated velocity plot frame {i + 1}/{num_frames} ...")

        if writer is not None:
            writer.release()
        plt.close(fig_anim)
        self.log(f"[INFO] Finished animated velocity vs time video: {anim_path}")

        # -------- Rolling-average speed (if enabled) --------
        if window_sec > 0.0 and max_t > 0.0:
            self._make_rolling_average_outputs(
                track_data, out_dir, out_base, max_t, max_s, window_sec
            )

    # ------------------------------------------------------------------
    # Rolling-average speed plots (static + animated)
    # ------------------------------------------------------------------
    def _make_rolling_average_outputs(
        self,
        track_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
        out_dir: Path,
        out_base: str,
        max_t: float,
        max_s_raw: float,
        window_sec: float,
    ):
        """
        Compute and output rolling-average speed vs time, both static
        and animated. Rolling average is taken over all samples in the
        time window [t - window_sec, t] across all tracks.
        """
        self.log(
            f"[INFO] Computing rolling-average speed with window = {window_sec:.2f} s"
        )

        # Build a time grid for the rolling average
        fps_anim = 30.0
        num_points = int(max(60, min(600, max_t * fps_anim))) if max_t > 0 else 60
        tau_values = np.linspace(0.0, max_t, num_points)

        ra_values = []
        for tau in tau_values:
            t_start = max(0.0, tau - window_sec)
            speeds_window = []

            for t_rel, speeds in track_data.values():
                mask = (t_rel >= t_start) & (t_rel <= tau)
                if np.any(mask):
                    speeds_window.append(speeds[mask])

            if speeds_window:
                all_speeds = np.concatenate(speeds_window)
                ra_values.append(float(all_speeds.mean()))
            else:
                ra_values.append(np.nan)

        ra_values = np.array(ra_values, dtype=float)
        if np.all(np.isnan(ra_values)):
            self.log("[WARN] Rolling-average window produced no valid samples.")
            return

        # For y-limit, include both raw max and RA max for a consistent scale
        valid_ra = ra_values[~np.isnan(ra_values)]
        if valid_ra.size > 0:
            max_s_ra = float(valid_ra.max())
        else:
            max_s_ra = 0.0

        y_max = max(max_s_raw, max_s_ra, 1.0) * 1.05

        # ---- Static rolling-average plot ----
        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
        ax.set_title(
            f"Rolling-average Speed (window = {window_sec:.2f} s)"
        )
        ax.set_xlabel("Time since track start [s]")
        ax.set_ylabel("Speed [px/s]")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)

        mask_valid = ~np.isnan(ra_values)
        ax.plot(
            tau_values[mask_valid],
            ra_values[mask_valid],
            color="C1",
            linewidth=2.0,
            label="Rolling avg speed",
        )

        ax.set_xlim(0.0, max_t * 1.05)
        ax.set_ylim(0.0, y_max)
        ax.legend(loc="upper right")

        out_static = out_dir / f"{out_base}_rolling_avg_speed.png"
        fig.tight_layout()
        fig.savefig(str(out_static), dpi=120)
        plt.close(fig)
        self.log(f"[INFO] Saved static rolling-average speed plot to: {out_static}")

        # ---- Animated rolling-average plot ----
        dpi_anim = 120
        fig_anim, ax_anim = plt.subplots(figsize=(8, 6), dpi=dpi_anim)
        ax_anim.set_title(
            f"Rolling-average Speed (animated, window = {window_sec:.2f} s)"
        )
        ax_anim.set_xlabel("Time since track start [s]")
        ax_anim.set_ylabel("Speed [px/s]")
        ax_anim.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
        ax_anim.set_xlim(0.0, max_t * 1.05)
        ax_anim.set_ylim(0.0, y_max)
        fig_anim.tight_layout()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        anim_path = out_dir / f"{out_base}_rolling_avg_speed_anim.mp4"
        writer = None

        self.log(f"[INFO] Writing animated rolling-average speed to: {anim_path}")

        num_frames = len(tau_values)
        for i, tau in enumerate(tau_values):
            ax_anim.cla()
            ax_anim.set_title(
                f"Rolling-average Speed (animated, window = {window_sec:.2f} s)"
            )
            ax_anim.set_xlabel("Time since track start [s]")
            ax_anim.set_ylabel("Speed [px/s]")
            ax_anim.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
            ax_anim.set_xlim(0.0, max_t * 1.05)
            ax_anim.set_ylim(0.0, y_max)

            # Plot RA curve up to current time
            mask_frame = (tau_values <= tau) & ~np.isnan(ra_values)
            if np.any(mask_frame):
                ax_anim.plot(
                    tau_values[mask_frame],
                    ra_values[mask_frame],
                    color="C1",
                    linewidth=2.0,
                    label="Rolling avg speed",
                )

            ax_anim.axvline(tau, color="k", linestyle="--", alpha=0.3)
            ax_anim.legend(loc="upper right")

            fig_anim.canvas.draw()
            buf = np.asarray(fig_anim.canvas.buffer_rgba())  # H x W x 4
            height_px, width_px, _ = buf.shape
            img_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)

            if writer is None:
                writer = cv2.VideoWriter(
                    str(anim_path), fourcc, fps_anim, (width_px, height_px)
                )

            writer.write(img_bgr)

            if (i + 1) % 20 == 0:
                self.log(
                    f"Animated rolling-average frame {i + 1}/{num_frames} ..."
                )

        if writer is not None:
            writer.release()
        plt.close(fig_anim)
        self.log(f"[INFO] Finished animated rolling-average speed video: {anim_path}")


def main():
    app = QApplication(sys.argv)
    win = PTVMakerWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
