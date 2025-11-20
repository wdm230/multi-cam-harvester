
# Multi-Camera Harvester GUI

This project is a small multi-camera GUI + backend for GigE / GenICam cameras using [Harvester](https://github.com/genicam/harvesters). It does:

- Live preview from one or more cameras  
- Recording to disk (video + snapshots)  
- ChArUco-based camera calibration  
- Optional simulated cameras for development  
- An optional live PTV (particle tracking) overlay  

It’s built around a single-camera backend (`HarvesterCameraManager` in `backend.py`) and a Qt GUI (`gui_multi.py`) that manages a grid of camera tiles.

---

## Hardware & camera support

### What cameras are supported?

Anything that:

- Is GenICam / GenTL compatible  
- Has a working GenTL **.cti** producer  
- Can be opened by Harvester  


If Harvester can see your camera in its `device_info_list`, this GUI should be able to stream it.

---

## What is the `.cti` file and why isn’t it in the repo?

Harvester talks to cameras via a **GenTL producer**. The producer is provided by the camera vendor (or a generic SDK) and is shipped as a **`.cti`** shared library.

Examples (paths will vary):

- Allied Vision / Mako: something like `VimbaGigETL.cti` from the VimbaX / Vimba SDK  
- Basler: a `.cti` from the Pylon SDK  
- Other vendors: look in their SDK install under `bin/`, `cti/`, or `transportlayers/`  


In `gui_multi.py` there is a constant near the top:

```python
CTI_PATH = "VimbaGigETL.cti"
````

Change this to the full path of your GenTL producer.


---

## Project layout (main pieces)

* `backend.py`
  Thin wrapper around Harvester for a **single camera**:

  * `HarvesterCameraManager(cti_path, device_index)`
  * `open()` → start acquisition for live preview
  * `add_file(session_dir, session_id, record_video, snapshot_fps, video_fps, codec)`
  * `start()` / `stop()` → start/stop recording
  * `fetch_next()` -> get frames for preview + recording
  * Handles video writing, snapshot timing, and recording FPS separately from preview FPS.

* `gui_multi.py`
  PyQt5 GUI that:

  * Creates multiple `HarvesterCameraManager` instances (one per real camera)
  * Optionally adds simulated cameras
  * Shows all cameras in a grid of tiles
  * Has toolbar buttons for:

    * Connect / Disconnect all
    * Start / Stop recording (all cameras)
    * Start / Stop PTV overlay
  * Manages a “current” camera for calibration, snapshots, etc.

* `calibration_panel.py`
  A ChArUco calibration dock:

  * Uses OpenCV’s aruco / Charuco routines
  * Lets you set board parameters (squares, marker size, dictionary, min corners)
  * Capture images from the active camera (timed or manual)
  * See a table of captured images and per-image errors
  * Run calibration and export a JSON file with intrinsics, distortion, etc.
  * Plots per-image RMS error with Matplotlib.

* `ptv.py`
  A small “live PTV” helper:

  * Detects bright particles on a dark background (threshold + contours)
  * Tracks up to `max_tracks` points per camera over time
  * Draws trajectories as polylines on top of the preview image
  * Has simple `start()` / `stop()` semantics:

    * **Start PTV** -> clear old traces and start a new tracking session
    * **Stop PTV** -> stop updating, but keep drawing existing traces

You’ll also have various calibration JSON outputs, PNG snapshots, etc.

---

## Running the GUI

1. Make sure your `.cti` path is correct in `gui_multi.py`:

   ```python
   CTI_PATH = r"C:\path\to\your\producer.cti"
   ENABLE_REAL_CAMERAS = True
   ```

   If you don’t have cameras handy, you can set `ENABLE_SIMULATED_CAMERAS = True` to play with the GUI.

2. Start the GUI:

   ```bash
   python gui_multi.py
   ```

3. Use the toolbar:

   * **Connect all**
     Opens all configured real cameras, starts acquisition, preview goes live.
     (Sim cameras also start if enabled.)

   * **Start all recording / Stop all recording**
     Starts and stops recording across all cameras. Recording FPS is independent of preview FPS.

   * **Start PTV / Stop PTV**
     Turns the live PTV overlay on/off for all cameras:

     * Starting PTV clears old traces and begins tracking new ones.
     * Stopping PTV freezes the existing trajectories (they remain drawn on subsequent frames).

---

## Calibration workflow (ChArUco)

The calibration dock (`CalibrationPanel`) plugs into the active camera in the GUI.

Typical flow:

1. **Configure board**

   * Set squares in X/Y
   * Set square/marker sizes (in mm)
   * Choose the ArUco dictionary
   * Choose a minimum corner count to reject bad frames

2. **Capture images**

   * Choose “Timed” or “Manual” mode
   * Set interval and maximum images if using timed capture
   * Move the board around the field of view:

     * Center, corners, various distances, some tilt, etc.
   * You should see rows show up in **Captured Images** with corner counts.

3. **Run calibration**

   * Make sure “Used” is checked for the images you want in the solve
   * Click **Run calibration**
   * Panel shows:

     * Overall RMS error (in pixels)
     * Per-image error in the table and plot

4. **Export**

   * Use **Export JSON** to write out intrinsics and distortion coefficients for later use.


---

## Live PTV overlay

The PTV pipeline in `ptv.py` is intentionally simple, tuned for something like “small bright lights on dark water at night”:

* Thresholds the frame to pick out bright blobs
* Filters by area
* Computes centroids
* Associates centroids with existing tracks via nearest neighbor
* Draws polyline traces over time

Key behavior:

* Per-camera `PTVTracker` objects live alongside the camera backends.
* When you click **Start PTV**:

  * Each tracker’s `start()` is called
  * Tracks are cleared and new ones start accumulating
* When you click **Stop PTV**:

  * Each tracker’s `stop()` is called
  * No new detections or updates are made, but existing tracks are still drawn on every new frame

Internally, detection is tuned via `PTVConfig`:

* `threshold` – binary threshold for bright particles
* `min_area` – reject tiny specks
* `max_tracks` – cap on number of trajectories
* `max_history` – max number of points stored per track
* `max_dist_px` – maximum jump allowed between frames
* `process_every_n_frames` – you can skip frames to reduce CPU load
* `downscale_factor` – run detection on downscaled frames and rescale centroids

If PTV ever feels heavy, you can adjust `process_every_n_frames` and `downscale_factor` to trade resolution/FPS for CPU.

---

## Notes / limitations

* Assumes Mono8 images; if you switch your camera to color formats, you’ll need to adapt the backend.
* The PTV overlay is meant as a visual tool rather than a full scientific PTV library, but you can pull the track data out of `PTVTracker.tracks` if you want to log trajectories.


