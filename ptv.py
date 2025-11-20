"""
ptv.py - simple particle tracking velocimetry (PTV) helper.

Usage pattern (per camera):

    from ptv import PTVConfig, PTVTracker

    tracker = PTVTracker(config=PTVConfig())

    # When user clicks "Start PTV":
    tracker.start()  # clears old traces and begins a new session

    # When user clicks "Stop PTV":
    tracker.stop()   # stops updating, but traces remain and are drawn

    # Each preview frame:
    frame_with_traces = tracker.process_frame(frame)

    # If you want to wipe old traces (e.g. when starting a new experiment)
    tracker.clear()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import cv2
import numpy as np


Point = Tuple[float, float]


@dataclass
class PTVConfig:
    """
    Configuration for a PTVTracker.
    """

    # Bright particle on dark background: threshold near the top
    threshold: int = 200

    # Minimum area (in pixels) for a blob to be considered a particle
    min_area: float = 5.0

    # Maximum number of concurrent tracks
    max_tracks: int = 30

    # Maximum number of points stored per track
    max_history: int = 5000

    # Maximum allowed jump (in pixels) between frames to be considered the same track
    max_dist_px: float = 50.0

    # How many "missed" frames before a track is dropped (only while active)
    max_missed_frames: int = 30

    # Process PTV on every Nth frame (1 = every frame, 2 = every other frame, etc.)
    process_every_n_frames: int = 1

    # Optional downscale factor for detection (0.5 = half-res for speed)
    # Detected centroids are rescaled back up to the original coordinate system.
    downscale_factor: float = 1.0


@dataclass
class Track:
    id: int
    points: List[Point] = field(default_factory=list)
    last_pos: Optional[Point] = None
    missed_frames: int = 0  # how many frames since last update


class PTVTracker:
    """
    Simple PTV tracker for one camera.

    .start()  -> begin a new PTV session (clears old tracks)
    .stop()   -> stop updating tracks but keep drawing existing traces
    .clear()  -> clear tracks entirely
    .process_frame(frame) -> returns a frame with tracks drawn
    """

    def __init__(self, config: Optional[PTVConfig] = None):
        self.config = config or PTVConfig()
        self.active: bool = False
        self.tracks: List[Track] = []
        self._next_id: int = 0
        self._frame_idx: int = 0

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------
    def start(self) -> None:
        """
        Start a new PTV session.

        This clears any existing tracks/traces and begins tracking anew.
        """
        self.tracks.clear()
        self._next_id = 0
        self._frame_idx = 0
        self.active = True

    def stop(self) -> None:
        """
        Stop updating tracks.

        Existing tracks/traces are preserved and continue to be drawn,
        but no new detections or track updates are performed.
        """
        self.active = False

    def clear(self) -> None:
        """
        Clear all tracks/traces but do not change active/inactive state.
        """
        self.tracks.clear()
        self._next_id = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray, copy: bool = True) -> np.ndarray:
        """
        Process a single frame and return a frame with tracks drawn.
        """
        self._frame_idx += 1

        if copy:
            out = frame.copy()
        else:
            out = frame

        # Ensure we have a 3-channel BGR frame for drawing
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        elif out.ndim == 3 and out.shape[2] == 1:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        if self.active:
            # Optional frame skipping for performance
            if self.config.process_every_n_frames <= 1 or (
                self._frame_idx % self.config.process_every_n_frames == 0
            ):
                centroids = self._detect_centroids(frame)
                self._update_tracks(centroids)
                self._prune_stale_tracks()

        # Draw tracks regardless of active/inactive, so traces persist
        self._draw_tracks(out)
        return out


    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def _detect_centroids(self, frame: np.ndarray) -> List[Point]:
        """
        Detect bright particles on a dark background and return centroids
        in the original image coordinate system.
        """
        # Convert to grayscale if needed
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2:
            gray = frame
        else:
            # Unexpected format; bail with no detections
            return []

        f = max(self.config.downscale_factor, 1e-3)  # avoid zero
        if 0.0 < f < 1.0:
            h, w = gray.shape[:2]
            small_w = max(1, int(w * f))
            small_h = max(1, int(h * f))
            small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        else:
            small = gray
            f = 1.0  # effectively no scaling

        # Threshold: bright particles become white blobs
        _, bw = cv2.threshold(
            small, self.config.threshold, 255, cv2.THRESH_BINARY
        )

        # Find contours (OpenCV 3/4 compatibility)
        cnts = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) == 2:
            contours = cnts[0]
        else:
            contours = cnts[1]

        centroids: List[Point] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.config.min_area:
                continue
            M = cv2.moments(c)
            if M["m00"] <= 0:
                continue
            cx_small = M["m10"] / M["m00"]
            cy_small = M["m01"] / M["m00"]

            # Rescale centroid to original image coordinates
            cx = cx_small / f
            cy = cy_small / f
            centroids.append((cx, cy))

        return centroids

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------
    def _update_tracks(self, centroids: List[Point]) -> None:
        """
        Associate detected centroids with existing tracks, or start new ones.
        """
        used = set()
        max_d2 = self.config.max_dist_px * self.config.max_dist_px

        # First, try to extend existing tracks
        for track in self.tracks:
            if track.last_pos is None:
                track.missed_frames += 1
                continue

            px, py = track.last_pos
            best_j = None
            best_d2 = max_d2

            for j, (cx, cy) in enumerate(centroids):
                if j in used:
                    continue
                dx = cx - px
                dy = cy - py
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best_j = j

            if best_j is not None:
                cx, cy = centroids[best_j]
                track.points.append((cx, cy))
                track.last_pos = (cx, cy)
                track.missed_frames = 0
                used.add(best_j)
            else:
                track.missed_frames += 1

            # Limit history length
            if len(track.points) > self.config.max_history:
                track.points = track.points[-self.config.max_history :]

        # Then, start new tracks for any unused centroids, up to max_tracks
        for j, (cx, cy) in enumerate(centroids):
            if j in used:
                continue
            if len(self.tracks) >= self.config.max_tracks:
                break
            t = Track(
                id=self._next_id,
                points=[(cx, cy)],
                last_pos=(cx, cy),
                missed_frames=0,
            )
            self.tracks.append(t)
            self._next_id += 1

    def _prune_stale_tracks(self) -> None:
        """
        Drop tracks that haven't been updated in a while.
        Only runs while active, so traces persist when stopped.
        """
        keep: List[Track] = []
        for t in self.tracks:
            if t.missed_frames <= self.config.max_missed_frames:
                keep.append(t)
        self.tracks = keep

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _draw_tracks(self, frame_bgr: np.ndarray) -> None:
        """
        Draw track polylines on the given frame (in-place).
        Assumes frame_bgr is already 3-channel BGR.
        """
        for t in self.tracks:
            if len(t.points) < 2:
                continue
            pts = np.array(t.points, dtype=np.int32).reshape(-1, 1, 2)
            # Bright green polyline
            cv2.polylines(
                frame_bgr, [pts],
                isClosed=False,
                color=(0, 255, 0),
                thickness=2,
            )

