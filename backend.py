# backend.py
"""
Backend for Mako / GigE camera using Harvester.

Public API:

    cm = HarvesterCameraManager(cti_path=...)
    cm.open()   # starts acquisition immediately → live preview available

    # When you want to record:
    cm.add_file(session_dir, session_id, record_video=True, snapshot_fps=1.0)
    cm.start()  # start *recording* (stream was already live)

    while cm.streaming:
        frame = cm.fetch_next(timeout=0.5)
        ...

    cm.stop()   # stop recording (preview still live)
    cm.close()  # stop acquisition + cleanup
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time  # <-- NEW

import cv2
import numpy as np
from harvesters.core import Harvester
from genicam.gentl import TimeoutException


@dataclass
class SessionConfig:
    session_dir: Path
    session_id: str
    record_video: bool = False
    video_fps: float = 30.0
    video_codec: str = "XVID"  # NEW
    # NEW: snapshots based on wall-clock FPS instead of every N frames
    snapshot_fps: float = 0.0  # 0.0 = disabled



class HarvesterCameraManager:
    """
    Thin backend around Harvester for a single camera.

    Streaming starts on open():
        cm = HarvesterCameraManager(cti_path, device_index=0)
        cm.open()   # acquisition starts → frames available for preview

    Recording is controlled by add_file() + start() / stop():
        cm.add_file("captures/test1", "test1", record_video=True, snapshot_fps=1.0)
        cm.start()   # start recording
        ...
        cm.stop()    # stop recording (preview continues)
    """

    def __init__(self, cti_path: str, device_index: int = 0):
        self._cti_path = str(cti_path)
        self._device_index = device_index

        self._h: Optional[Harvester] = None
        self._ia = None  # ImageAcquirer

        # acquisition vs recording
        self._acq_started: bool = False       # acquisition on/off (live preview)
        self._session_cfg: Optional[SessionConfig] = None
        self._pending_cfg: Optional[SessionConfig] = None
        self._recording: bool = False         # whether we are writing video/snapshots

        self._frame_index = 0
        self._writer: Optional[cv2.VideoWriter] = None

        # NEW: track last snapshot time for snapshot_fps logic
        self._last_snapshot_time: Optional[float] = None

    # ---- lifetime management ------------------------------------------------

    def open(self):
        """
        Load CTI and create an ImageAcquirer, set PixelFormat to Mono8,
        and start acquisition for live streaming.
        """
        if self._h is not None:
            return  # already open

        h = Harvester()
        # Newer Harvester API: add_file() / create()
        h.add_file(self._cti_path)
        h.update()

        if not h.device_info_list:
            h.reset()
            raise RuntimeError("No cameras found via GenTL/CTI.")

        ia = h.create(self._device_index)

        # Force a known pixel format (Mono8) so display & recording behave.
        try:
            remote = ia.remote_device.node_map
            pf = remote.PixelFormat
            if "Mono8" in pf.symbolics:
                pf.value = "Mono8"
        except Exception:
            # If this fails, we'll just use whatever the camera is already set to.
            pass

        self._h = h
        self._ia = ia

        # Start acquisition immediately for live preview
        try:
            self._ia.start()
            self._acq_started = True
        except Exception as e:
            # If we can't start acquisition, cleanup and raise
            self._ia.destroy()
            self._h.reset()
            self._ia = None
            self._h = None
            raise RuntimeError(f"Failed to start acquisition: {e}")

    def close(self):
        """Stop acquisition, destroy ImageAcquirer and reset Harvester."""
        # stop any active recording
        self.stop()

        if self._ia is not None:
            try:
                if self._acq_started:
                    self._ia.stop()
            except Exception:
                pass
            try:
                self._ia.destroy()
            except Exception:
                pass
            self._ia = None

        if self._h is not None:
            try:
                self._h.reset()
            except Exception:
                pass
            self._h = None

        self._acq_started = False

    # ---- "add_file / start / stop" public API ------------------------------

    def add_file(
        self,
        session_dir: Path | str,
        session_id: str,
        *,
        record_video: bool = True,
        video_fps: float = 30.0,
        video_codec: str = "XVID",      # NEW
        snapshot_fps: float = 0.0,      # NEW: snapshots per second
    ) -> SessionConfig:
        """
        Configure where to write video/snapshots, but do NOT alter acquisition.

        Call start() afterwards to actually begin recording.

        snapshot_fps:
            0.0 → no snapshots
            1.0 → ~1 snapshot per second
            10.0 → up to ~10 snapshots per second
        """
        if self._recording:
            raise RuntimeError(
                "Cannot add_file() while a recording is running; call stop() first."
            )

        session_dir = Path(session_dir)
        cfg = SessionConfig(
            session_dir=session_dir,
            session_id=session_id,
            record_video=record_video,
            video_fps=video_fps,
            video_codec=video_codec,   # NEW
            snapshot_fps=snapshot_fps,
        )
        self._pending_cfg = cfg
        return cfg


    def start(self):
        """
        Start recording using the pending SessionConfig from add_file().

        Acquisition is assumed to already be running from open().
        """
        if self._ia is None:
            raise RuntimeError("Camera not opened; call open() first.")

        if not self._acq_started:
            # Shouldn't normally happen, but we can try to recover:
            self._ia.start()
            self._acq_started = True

        if self._recording:
            raise RuntimeError("Recording already running; call stop() first.")

        if self._pending_cfg is None:
            raise RuntimeError("No session configured; call add_file() before start().")

        self.start_session(self._pending_cfg)
        self._pending_cfg = None

    def stop(self):
        """Stop the current recording session, if any (preview stays live)."""
        self.stop_session()

    # ---- lower-level start/stop (recording only) ---------------------------

    def start_session(self, cfg: SessionConfig):
        """
        Lower-level start: direct SessionConfig.

        Used by start() and can also be called manually if you want.
        Does NOT start/stop acquisition; only controls recording/snapshots.
        """
        if self._ia is None:
            raise RuntimeError("Camera not opened; call open() first.")

        if not self._acq_started:
            raise RuntimeError("Acquisition not started; open() should start it.")

        if self._recording:
            raise RuntimeError("Recording already running; call stop_session() first.")

        cfg.session_dir.mkdir(parents=True, exist_ok=True)

        self._session_cfg = cfg
        self._frame_index = 0
        self._writer = None
        self._recording = True
        self._last_snapshot_time = None  # NEW: reset when session starts

    def stop_session(self):
        """Stop recording and finalize video writer if any (preview is unaffected)."""
        if not self._recording:
            return

        # Close writer if we had one
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
            self._writer = None

        self._recording = False
        self._session_cfg = None
        self._last_snapshot_time = None

    # ---- acquisition --------------------------------------------------------

    @property
    def streaming(self) -> bool:
        """True if acquisition is active (i.e., preview can fetch frames)."""
        return self._acq_started and self._ia is not None

    @property
    def recording(self) -> bool:
        """True if we are currently writing video/snapshots."""
        return self._recording

    def _ensure_writer(self, img: np.ndarray):
        """Lazy-init video writer based on first frame shape."""
        if not self._recording:
            return
        if self._writer is not None:
            return
        if self._session_cfg is None:
            return
        if not self._session_cfg.record_video:
            return

        height, width = img.shape[:2]

        # Choose extension + fourcc based on codec
        codec = (self._session_cfg.video_codec or "XVID").upper()

        if codec == "MP4V":
            ext = ".mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        elif codec == "MJPG":
            ext = ".avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        else:
            codec = "XVID"
            ext = ".avi"
            fourcc = cv2.VideoWriter_fourcc(*"XVID")

        video_path = self._session_cfg.session_dir / f"{self._session_cfg.session_id}{ext}"

        self._writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            float(self._session_cfg.video_fps),
            (width, height),
            isColor=img.ndim == 3 and img.shape[2] == 3,
        )


    def fetch_next(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        """
        Fetch next frame as a numpy array (Mono8 assumed for the Mako G-131B).

        Returns None on timeout or if acquisition not running.
        """
        if not self.streaming:
            return None

        try:
            # Use fetch() not try_fetch(), catch TimeoutException
            buffer = self._ia.fetch(timeout=timeout)
        except TimeoutException:
            return None
        except Exception:
            # Any other GenTL error -> signal caller by returning None
            return None

        if buffer is None:
            return None

        try:
            comp = buffer.payload.components[0]
            img = comp.data.reshape(comp.height, comp.width)

            # Ensure contiguous and DETACH from Harvester buffer
            img = np.ascontiguousarray(img).copy()

            # Lazy-init writer
            self._ensure_writer(img)

            # Write video if we're recording
            if self._writer is not None:
                try:
                    self._writer.write(img)
                except Exception:
                    # Don't kill preview if writer glitches
                    pass

            # Snapshots (time-based, using snapshot_fps)
            if self._recording and self._session_cfg:
                self._frame_index += 1
                fps = float(self._session_cfg.snapshot_fps)
                if fps > 0.0:
                    now = time.time()
                    interval = 1.0 / fps
                    if self._last_snapshot_time is None or (now - self._last_snapshot_time) >= interval:
                        self._last_snapshot_time = now
                        snap_path = (
                            self._session_cfg.session_dir
                            / f"{self._session_cfg.session_id}_frame_{self._frame_index:06d}.png"
                        )
                        try:
                            cv2.imwrite(str(snap_path), img)
                        except Exception:
                            pass

            return img
        finally:
            # Queue buffer back to the producer for reuse
            try:
                buffer.queue()
            except Exception:
                pass
    # ---- Camera parameter accessors ----------------------------------------

    def _remote(self):
        """
        Convenience accessor for the GenICam node map of the remote device.
        """
        if self._ia is None:
            raise RuntimeError("Camera not opened; call open() first.")
        return self._ia.remote_device.node_map

    # --- Gain ---------------------------------------------------------------

    def get_gain(self) -> float:
        remote = self._remote()
        # Try common node names
        for name in ("Gain", "GainRaw"):
            if hasattr(remote, name):
                node = getattr(remote, name)
                return float(node.value)
        raise AttributeError("Gain node not found on camera.")

    def set_gain(self, value: float) -> None:
        remote = self._remote()
        for name in ("Gain", "GainRaw"):
            if hasattr(remote, name):
                node = getattr(remote, name)
                node.value = value
                return
        raise AttributeError("Gain node not found on camera.")

    # --- Exposure -----------------------------------------------------------

    def get_exposure(self) -> float:
        """
        Returns exposure in whatever units the camera node uses
        (often microseconds for ExposureTime*).
        """
        remote = self._remote()
        for name in ("ExposureTime", "ExposureTimeAbs", "ExposureTimeRaw"):
            if hasattr(remote, name):
                node = getattr(remote, name)
                return float(node.value)
        raise AttributeError("Exposure node not found on camera.")

    def set_exposure(self, value: float) -> None:
        remote = self._remote()
        for name in ("ExposureTime", "ExposureTimeAbs", "ExposureTimeRaw"):
            if hasattr(remote, name):
                node = getattr(remote, name)
                node.value = value
                return
        raise AttributeError("Exposure node not found on camera.")

    # --- Auto white balance -------------------------------------------------

    def get_white_balance_auto(self) -> bool:
        """
        Returns True if auto white balance appears to be enabled.
        """
        remote = self._remote()
        for name in ("BalanceWhiteAuto", "WhiteBalanceAuto"):
            if hasattr(remote, name):
                node = getattr(remote, name)
                val = str(node.value)
                # Typical enum values: Off, Once, Continuous
                return val.lower() not in ("off", "0")
        raise AttributeError("White balance auto node not found on camera.")

    def set_white_balance_auto(self, enabled: bool) -> None:
        remote = self._remote()
        for name in ("BalanceWhiteAuto", "WhiteBalanceAuto"):
            if hasattr(remote, name):
                node = getattr(remote, name)
                if enabled:
                    # Prefer Continuous, fall back to Once
                    if "Continuous" in node.symbolics:
                        node.value = "Continuous"
                    elif "Once" in node.symbolics:
                        node.value = "Once"
                    else:
                        # last resort: any non-Off value
                        for sym in node.symbolics:
                            if sym.lower() != "off":
                                node.value = sym
                                break
                else:
                    if "Off" in node.symbolics:
                        node.value = "Off"
                    else:
                        # If no Off, just pick first symbol
                        node.value = node.symbolics[0]
                return
        raise AttributeError("White balance auto node not found on camera.")
