import json
import datetime

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QFileDialog,
    QMessageBox,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from pathlib import Path


class CalibrationWorker(QThread):
    """
    Runs ChArUco calibration in a background thread.

    Emits:
        result_ready(dict): {rms, camera_matrix, dist_coeffs, per_image_errors}
        error(str): error message
    """
    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, board, all_corners, all_ids, image_size, parent=None):
        super().__init__(parent)
        self._board = board
        self._all_corners = all_corners
        self._all_ids = all_ids
        self._image_size = image_size

    def run(self):
        import cv2
        import cv2.aruco as aruco
        import numpy as np

        try:
            # Prefer Extended if available (gives per-view errors)
            if hasattr(aruco, "calibrateCameraCharucoExtended"):
                ret, K, D, rvecs, tvecs, stdInt, stdExt, perViewErrors = (
                    aruco.calibrateCameraCharucoExtended(
                        charucoCorners=self._all_corners,
                        charucoIds=self._all_ids,
                        board=self._board,
                        imageSize=self._image_size,
                        cameraMatrix=None,
                        distCoeffs=None,
                    )
                )

                # perViewErrors is usually Nx1 or (N,), make it a flat list of floats
                perViewErrors = np.array(perViewErrors).reshape(-1)
                per_image_errors = [float(e) for e in perViewErrors]
            else:
                # Older bindings: no per-view errors, just a single RMS
                ret, K, D, rvecs, tvecs = aruco.calibrateCameraCharuco(
                    charucoCorners=self._all_corners,
                    charucoIds=self._all_ids,
                    board=self._board,
                    imageSize=self._image_size,
                    cameraMatrix=None,
                    distCoeffs=None,
                )
                # Best we can do: same RMS for each used image
                per_image_errors = [float(ret)] * len(self._all_corners)

            self.result_ready.emit(
                {
                    "rms": float(ret),
                    "camera_matrix": K,
                    "dist_coeffs": D,
                    "per_image_errors": per_image_errors,
                }
            )
        except Exception as e:
            self.error.emit(str(e))



class CalibrationPanel(QWidget):
    """
    MATLAB-like camera calibration panel using ChArUco.

    - Uses the latest frame from a callback (e.g. MainWindow.latest_frame).
    - Supports manual and timed capture modes.
    - Runs ChArUco calibration and shows per-image RMS errors in a bar plot.
    - Exports calibration parameters to JSON.
    """

    def __init__(self, parent, get_frame_callable):
        super().__init__(parent)
        self.get_frame = get_frame_callable  # function returning np.ndarray or None

        # Internal state
        self._captures = []          # list of dicts: image_path, corners, ids, used, error
        self._dictionary = None
        self._board = None
        self._image_size = None      # (width, height)
        self._camera_matrix = None
        self._dist_coeffs = None
        self._rms = None

        # Timer for timed capture
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer_capture)
        self._calib_worker = None
        self._last_used_indices = []
        self._build_ui()

    # ------------------------------------------------------------------ UI ---

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Board settings --------------------------------------------------
        board_box = QGroupBox("ChArUco Board")
        board_form = QFormLayout()

        self.squares_x_spin = QSpinBox()
        self.squares_x_spin.setRange(2, 50)
        self.squares_x_spin.setValue(7)

        self.squares_y_spin = QSpinBox()
        self.squares_y_spin.setRange(2, 50)
        self.squares_y_spin.setValue(4)

        self.square_size_spin = QDoubleSpinBox()
        self.square_size_spin.setRange(0.1, 1000.0)
        self.square_size_spin.setDecimals(3)
        self.square_size_spin.setValue(25.4)  # mm
        self.square_size_spin.setSuffix(" mm")

        self.marker_size_spin = QDoubleSpinBox()
        self.marker_size_spin.setRange(0.1, 1000.0)
        self.marker_size_spin.setDecimals(3)
        self.marker_size_spin.setValue(20.32)  # mm
        self.marker_size_spin.setSuffix(" mm")

        self.dict_combo = QComboBox()
        # Include all the ones you actually use; default to 6x6
        self.dict_combo.addItems([
            "DICT_6X6_250",
            "DICT_6X6_1000",
            "DICT_4X4_50",
            "DICT_5X5_50",
            "DICT_5X5_100",
        ])
        self.dict_combo.setCurrentText("DICT_6X6_250")

        self.min_corners_spin = QSpinBox()
        self.min_corners_spin.setRange(4, 500)
        self.min_corners_spin.setValue(8)  # more forgiving default

        board_form.addRow("Squares (X):", self.squares_x_spin)
        board_form.addRow("Squares (Y):", self.squares_y_spin)
        board_form.addRow("Square size:", self.square_size_spin)
        board_form.addRow("Marker size:", self.marker_size_spin)
        board_form.addRow("Dictionary:", self.dict_combo)
        board_form.addRow("Min ChArUco corners:", self.min_corners_spin)

        board_box.setLayout(board_form)
        layout.addWidget(board_box)

        # --- Capture settings -----------------------------------------------
        capture_box = QGroupBox("Capture")
        capture_form = QFormLayout()

        self.capture_mode_combo = QComboBox()
        self.capture_mode_combo.addItems(["Timed", "Manual"])

        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.1, 10.0)
        self.interval_spin.setDecimals(2)
        self.interval_spin.setValue(1.0)
        self.interval_spin.setSuffix(" s")

        self.max_images_spin = QSpinBox()
        self.max_images_spin.setRange(0, 1000)
        self.max_images_spin.setValue(0)  # 0 = unlimited

        btn_row = QHBoxLayout()
        self.start_capture_btn = QPushButton("Start capture")
        self.stop_capture_btn = QPushButton("Stop")
        self.capture_now_btn = QPushButton("Capture now")

        self.stop_capture_btn.setEnabled(False)

        btn_row.addWidget(self.start_capture_btn)
        btn_row.addWidget(self.stop_capture_btn)
        btn_row.addWidget(self.capture_now_btn)

        capture_form.addRow("Mode:", self.capture_mode_combo)
        capture_form.addRow("Interval:", self.interval_spin)
        capture_form.addRow("Max images (0 = inf):", self.max_images_spin)
        capture_form.addRow("", QWidget())  # spacer
        capture_form.addRow("", btn_row)

        self.status_label = QLabel("Ready.")
        capture_form.addRow("Status:", self.status_label)

        capture_box.setLayout(capture_form)
        layout.addWidget(capture_box)

        # Connect capture buttons
        self.start_capture_btn.clicked.connect(self._on_start_capture)
        self.stop_capture_btn.clicked.connect(self._on_stop_capture)
        self.capture_now_btn.clicked.connect(self._on_capture_now)

        # --- Captured images table ------------------------------------------
        table_box = QGroupBox("Captured Images")
        table_layout = QVBoxLayout()

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", "Corners", "Used", "Error (px)"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)

        table_buttons = QHBoxLayout()
        self.import_btn = QPushButton("Import folder...")
        self.delete_selected_btn = QPushButton("Delete selected")
        self.clear_all_btn = QPushButton("Clear all")

        table_buttons.addWidget(self.import_btn)
        table_buttons.addStretch(1)
        table_buttons.addWidget(self.delete_selected_btn)
        table_buttons.addWidget(self.clear_all_btn)

        self.import_btn.clicked.connect(self._on_import_folder)
        self.delete_selected_btn.clicked.connect(self._on_delete_selected)
        self.clear_all_btn.clicked.connect(self._on_clear_all)

        table_layout.addWidget(self.table)
        table_layout.addLayout(table_buttons)
        table_box.setLayout(table_layout)
        layout.addWidget(table_box)

        # --- Calibration & results ------------------------------------------
        calib_box = QGroupBox("Calibration")
        calib_layout = QVBoxLayout()

        self.calibrate_btn = QPushButton("Run calibration")
        self.export_btn = QPushButton("Export JSON")
        self.export_btn.setEnabled(False)

        self.summary_label = QLabel("No calibration run yet.")

        # Matplotlib canvas for per-image errors
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        self.canvas = FigureCanvas(self.fig)

        btn_row2 = QHBoxLayout()
        btn_row2.addWidget(self.calibrate_btn)
        btn_row2.addWidget(self.export_btn)

        calib_layout.addLayout(btn_row2)
        calib_layout.addWidget(self.summary_label)
        calib_layout.addWidget(self.canvas)

        calib_box.setLayout(calib_layout)
        layout.addWidget(calib_box)

        self.calibrate_btn.clicked.connect(self._on_calibrate)
        self.export_btn.clicked.connect(self._on_export)

        layout.addStretch(1)

    # --------------------------------------------------------- board setup ---

    def _get_dictionary(self):
        # Map combo text to actual cv2.aruco dictionary
        import cv2.aruco as aruco

        name = self.dict_combo.currentText()
        mapping = {
            "DICT_4X4_50":   aruco.DICT_4X4_50,
            "DICT_5X5_50":   aruco.DICT_5X5_50,
            "DICT_5X5_100":  aruco.DICT_5X5_100,
            "DICT_6X6_250":  aruco.DICT_6X6_250,
            "DICT_6X6_1000": aruco.DICT_6X6_1000,
        }
        return aruco.getPredefinedDictionary(mapping[name])

    def _update_board(self):
        import cv2.aruco as aruco

        squares_x = self.squares_x_spin.value()
        squares_y = self.squares_y_spin.value()
        square_size_mm = self.square_size_spin.value()
        marker_size_mm = self.marker_size_spin.value()

        # Use meters internally
        square_len = square_size_mm / 1000.0
        marker_len = marker_size_mm / 1000.0

        self._dictionary = self._get_dictionary()

        # Try the common APIs in order
        if hasattr(aruco, "CharucoBoard_create"):
            self._board = aruco.CharucoBoard_create(
                squares_x, squares_y, square_len, marker_len, self._dictionary
            )
        elif hasattr(aruco, "CharucoBoard"):
            self._board = aruco.CharucoBoard(
                (squares_x, squares_y),
                square_len,
                marker_len,
                self._dictionary,
            )
        else:
            raise RuntimeError(
                "Your OpenCV build does not provide cv2.aruco.CharucoBoard; "
                "install opencv-contrib-python (and make sure no old OpenCV "
                "packages are shadowing it)."
            )

    def _ensure_board(self):
        if self._board is None or self._dictionary is None:
            self._update_board()

    # -------------------------------------------------------- capture flow ---

    def _on_start_capture(self):
        mode = self.capture_mode_combo.currentText()
        if mode == "Manual":
            self._timer.stop()
            self.start_capture_btn.setEnabled(False)
            self.stop_capture_btn.setEnabled(True)
            self.status_label.setText("Manual mode: press 'Capture now' to add images.")
        else:
            interval_s = self.interval_spin.value()
            self._timer.start(int(interval_s * 1000))
            self.start_capture_btn.setEnabled(False)
            self.stop_capture_btn.setEnabled(True)
            self.status_label.setText(f"Timed mode: capturing every {interval_s:.2f} s.")

    def _on_stop_capture(self):
        self._timer.stop()
        self.start_capture_btn.setEnabled(True)
        self.stop_capture_btn.setEnabled(False)
        self.status_label.setText("Capture stopped.")

    def _on_capture_now(self):
        ok = self._capture_one()
        if ok:
            self.status_label.setText(f"Captured {len(self._captures)} images.")
        else:
            # _capture_one sets a more specific message; keep a generic fallback
            if self.status_label.text() == "Ready.":
                self.status_label.setText("Capture failed (no pattern / no frame).")

    def _on_timer_capture(self):
        max_images = self.max_images_spin.value()
        if max_images > 0 and len(self._captures) >= max_images:
            self._timer.stop()
            self.start_capture_btn.setEnabled(True)
            self.stop_capture_btn.setEnabled(False)
            self.status_label.setText("Reached max images; capture stopped.")
            return

        ok = self._capture_one()
        if ok:
            self.status_label.setText(
                f"Timed capture: {len(self._captures)} images."
            )

    def _aruco_params(self):
        # Shared detection parameters (slightly looser than defaults)
        import cv2.aruco as aruco

        # Handle both old and new OpenCV APIs
        if hasattr(aruco, "DetectorParameters_create"):
            params = aruco.DetectorParameters_create()
        else:
            params = aruco.DetectorParameters()

        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 10
        params.minMarkerPerimeterRate = 0.02
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.03
        if hasattr(aruco, "CORNER_REFINE_SUBPIX"):
            params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

        return params


    def _capture_one(self) -> bool:
        frame = self.get_frame()
        if frame is None:
            QMessageBox.warning(self, "No frame", "No current frame available from camera.")
            return False

        # Ensure grayscale
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        h, w = gray.shape[:2]
        if self._image_size is None:
            self._image_size = (w, h)
        else:
            if self._image_size != (w, h):
                QMessageBox.warning(
                    self,
                    "Size mismatch",
                    f"Captured image size {w}x{h} does not match previous {self._image_size}. "
                    "Ignoring this frame."
                )
                return False

        self._ensure_board()
        import cv2.aruco as aruco

        params = self._aruco_params()
        corners, ids, _ = aruco.detectMarkers(gray, self._dictionary, parameters=params)

        if ids is None or len(ids) == 0:
            self.status_label.setText("No ArUco markers detected in this frame.")
            return False

        num, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            corners, ids, gray, self._board
        )

        min_req = self.min_corners_spin.value()
        if (
            charuco_corners is None
            or charuco_ids is None
            or num < min_req
        ):
            self.status_label.setText(
                f"Detected {num if charuco_corners is not None else 0} ChArUco corners "
                f"(min required: {min_req}) – frame skipped."
            )
            return False

        # ---------------------------------------------------------------------
        # Save this capture to disk under <current_save_dir>/calibrationSnapshots
        # ---------------------------------------------------------------------
        parent = self.parent()
        base_path = None

        if hasattr(parent, "dir_edit"):
            txt = parent.dir_edit.text().strip()
            base_path = txt or getattr(parent, "save_path", None)
        elif hasattr(parent, "save_path"):
            base_path = parent.save_path

        if not base_path:
            base_path = "captures"

        base_dir = Path(base_path)
        calib_dir = base_dir / "calibrationSnapshots"
        calib_dir.mkdir(parents=True, exist_ok=True)

        snap_idx = len(self._captures)
        filename = f"charuco_{snap_idx:04d}.png"
        file_path = calib_dir / filename

        try:
            cv2.imwrite(str(file_path), gray)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Save failed",
                f"Could not save calibration snapshot:\n{file_path}\n\n{e}",
            )
            return False

        # Store only metadata + corner data in memory, not full image arrays
        capture = {
            "image_path": str(file_path),
            "charuco_corners": charuco_corners.copy(),
            "charuco_ids": charuco_ids.copy(),
            "used": True,
            "error": None,
        }
        self._captures.append(capture)
        self._append_capture_row(capture)
        return True

    # ------------------------------------------------------ import folder ----

    def _on_import_folder(self):
        """
        Let the user choose a directory of images (e.g. calibrationSnapshots)
        and import them as ChArUco captures.
        """
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select folder with calibration images",
            "",
        )
        if not dir_path:
            return

        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            QMessageBox.warning(self, "Invalid folder", "Selected path is not a directory.")
            return

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        files = sorted(
            [p for p in dir_path.iterdir() if p.suffix.lower() in exts],
            key=lambda p: p.name,
        )
        if not files:
            QMessageBox.information(
                self,
                "No images",
                "No image files (.png, .jpg, .jpeg, .bmp, .tif, .tiff) found in this folder.",
            )
            return

        self._ensure_board()
        import cv2.aruco as aruco

        imported = 0
        skipped_size = 0
        skipped_nopattern = 0

        params = self._aruco_params()

        for path in files:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            h, w = img.shape[:2]
            if self._image_size is None:
                self._image_size = (w, h)
            else:
                if self._image_size != (w, h):
                    skipped_size += 1
                    continue

            corners, ids, _ = aruco.detectMarkers(img, self._dictionary, parameters=params)
            if ids is None or len(ids) == 0:
                skipped_nopattern += 1
                continue

            num, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners, ids, img, self._board
            )

            if (
                charuco_corners is None
                or charuco_ids is None
                or num < self.min_corners_spin.value()
            ):
                skipped_nopattern += 1
                continue

            capture = {
                "image_path": str(path),
                "charuco_corners": charuco_corners.copy(),
                "charuco_ids": charuco_ids.copy(),
                "used": True,
                "error": None,
            }
            self._captures.append(capture)
            self._append_capture_row(capture)
            imported += 1

        msg_lines = [f"Imported {imported} images from:\n{dir_path}"]
        if skipped_size:
            msg_lines.append(f"- Skipped {skipped_size} image(s) due to size mismatch.")
        if skipped_nopattern:
            msg_lines.append(f"- Skipped {skipped_nopattern} image(s) with no/insufficient ChArUco pattern.")

        QMessageBox.information(self, "Import complete", "\n".join(msg_lines))

        if imported > 0:
            self._camera_matrix = None
            self._dist_coeffs = None
            self._rms = None
            self.export_btn.setEnabled(False)
            self.summary_label.setText("Images changed; run calibration again.")

    # ---------------------------------------------------------- table mgmt ---

    def _append_capture_row(self, capture):
        row = self.table.rowCount()
        self.table.insertRow(row)

        idx_item = QTableWidgetItem(str(row))
        corners_item = QTableWidgetItem(str(len(capture["charuco_corners"])))
        used_item = QTableWidgetItem("Yes")
        used_item.setCheckState(Qt.Checked)
        error_item = QTableWidgetItem("")

        idx_item.setData(Qt.UserRole, row)
        corners_item.setData(Qt.UserRole, row)
        used_item.setData(Qt.UserRole, row)
        error_item.setData(Qt.UserRole, row)

        self.table.setItem(row, 0, idx_item)
        self.table.setItem(row, 1, corners_item)
        self.table.setItem(row, 2, used_item)
        self.table.setItem(row, 3, error_item)

    def _on_delete_selected(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            return

        for r in rows:
            if 0 <= r < len(self._captures):
                self._captures.pop(r)
            self.table.removeRow(r)

        for r in range(self.table.rowCount()):
            self.table.item(r, 0).setText(str(r))

        self._camera_matrix = None
        self._dist_coeffs = None
        self._rms = None
        self.export_btn.setEnabled(False)
        self.summary_label.setText("Images changed; run calibration again.")

    def _on_clear_all(self):
        self._captures.clear()
        self.table.setRowCount(0)
        self._camera_matrix = None
        self._dist_coeffs = None
        self._rms = None
        self.export_btn.setEnabled(False)
        self._image_size = None
        self.summary_label.setText("All captures cleared.")

    # -------------------------------------------------------- calibration ----

    def _on_calibration_result(self, result):
        """
        Called in the GUI thread when the worker finishes with valid results.
        """
        self._camera_matrix = result["camera_matrix"]
        self._dist_coeffs = result["dist_coeffs"]
        self._rms = result["rms"]
        per_image_errors = result["per_image_errors"]

        used_indices = self._last_used_indices or list(range(len(per_image_errors)))

        # Store back into captures
        for idx, img_err in zip(used_indices, per_image_errors):
            self._captures[idx]["error"] = float(img_err)

        # Update table + plot
        self._update_table_errors()
        self._update_error_plot(per_image_errors, used_indices)

        self.summary_label.setText(
            f"Calibration RMS error: {self._rms:.4f} px (using {len(used_indices)} images)."
        )

    def _on_calibration_error(self, msg):
        """
        Called in the GUI thread if the worker throws an exception.
        """
        QMessageBox.critical(self, "Calibration error", msg)
        self.summary_label.setText("Calibration failed.")

    def _on_calibration_thread_finished(self):
        """
        Called when the worker thread finishes (success or error).
        """
        self.calibrate_btn.setEnabled(True)
        if self._camera_matrix is not None:
            self.export_btn.setEnabled(True)

        self._calib_worker = None


    def _on_calibrate(self):
        if not self._captures:
            QMessageBox.information(self, "No images", "Capture some images first.")
            return

        # Build list of used captures
        all_corners = []
        all_ids = []
        used_indices = []

        for i, cap in enumerate(self._captures):
            item = self.table.item(i, 2)
            used = item.checkState() == Qt.Checked if item is not None else True

            if not used:
                self._captures[i]["used"] = False
                continue

            self._captures[i]["used"] = True
            all_corners.append(cap["charuco_corners"])
            all_ids.append(cap["charuco_ids"])
            used_indices.append(i)

        if len(all_corners) < 3:
            QMessageBox.warning(
                self,
                "Not enough images",
                "Need at least 3 good images to run calibration.",
            )
            return

        self._ensure_board()

        # Avoid starting two calibrations at once
        if self._calib_worker is not None:
            QMessageBox.information(
                self,
                "Calibration",
                "Calibration is already running.",
            )
            return

        # Save which images we used so we can map errors back later
        self._last_used_indices = used_indices

        # Update UI state
        self.summary_label.setText("Running calibration...")
        self.calibrate_btn.setEnabled(False)
        self.export_btn.setEnabled(False)

        # Start worker thread
        self._calib_worker = CalibrationWorker(
            board=self._board,
            all_corners=all_corners,
            all_ids=all_ids,
            image_size=self._image_size,
            parent=self,
        )
        self._calib_worker.result_ready.connect(self._on_calibration_result)
        self._calib_worker.error.connect(self._on_calibration_error)
        self._calib_worker.finished.connect(self._on_calibration_thread_finished)
        self._calib_worker.start()




    def _update_table_errors(self):
        if self._rms is None:
            return

        threshold = 2.0 * self._rms

        for i, cap in enumerate(self._captures):
            err = cap.get("error")
            item = self.table.item(i, 3)
            if err is None:
                if item is not None:
                    item.setText("")
                continue

            if item is None:
                item = QTableWidgetItem()
                self.table.setItem(i, 3, item)

            item.setText(f"{err:.4f}")
            if err > threshold:
                item.setForeground(Qt.red)
            else:
                item.setForeground(Qt.black)

    def _update_error_plot(self, per_image_errors, used_indices):
        self.ax.clear()
        if not per_image_errors:
            self.canvas.draw()
            return

        x = list(range(len(per_image_errors)))
        self.ax.bar(x, per_image_errors)
        self.ax.set_xlabel("Used image index")
        self.ax.set_ylabel("RMS error (px)")
        self.ax.set_title("Per-image reprojection error")
        self.fig.tight_layout()
        self.canvas.draw()

    # ----------------------------------------------------------- export JSON --

    def _on_export(self):
        if self._camera_matrix is None or self._dist_coeffs is None:
            QMessageBox.warning(self, "No calibration", "Run calibration first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export calibration JSON",
            "",
            "JSON files (*.json)",
        )
        if not path:
            return

        w, h = self._image_size
        K = self._camera_matrix
        D = self._dist_coeffs

        dist_list = [float(x) for x in D.flatten()]

        data = {
            "camera_id": "camera_0",  # you can customize with serial later
            "image_width": int(w),
            "image_height": int(h),
            "intrinsic_matrix": [
                [float(K[0, 0]), float(K[0, 1]), float(K[0, 2])],
                [float(K[1, 0]), float(K[1, 1]), float(K[1, 2])],
                [float(K[2, 0]), float(K[2, 1]), float(K[2, 2])],
            ],
            "distortion_model": "opencv_radial_tangential",
            "distortion_coeffs": dist_list,
            "reprojection_error_rms": float(self._rms),
            "per_image_error_rms": [
                float(cap["error"]) if cap.get("error") is not None else None
                for cap in self._captures
            ],
            "charuco": {
                "board_type": "charuco",
                "squares_x": self.squares_x_spin.value(),
                "squares_y": self.squares_y_spin.value(),
                "square_length_m": self.square_size_spin.value() / 1000.0,
                "marker_length_m": self.marker_size_spin.value() / 1000.0,
                "dictionary": self.dict_combo.currentText(),
                "units": "m",
            },
            "calibration_metadata": {
                "date": datetime.datetime.now().isoformat(),
                "software": "mako_gui_calibrator",
            },
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Export failed", f"Could not write JSON:\n{e}")
            return

        QMessageBox.information(self, "Export complete", f"Calibration saved to:\n{path}")
