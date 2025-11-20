import json
from pathlib import Path

import cv2
import numpy as np


def main():
    # --- hard-coded paths ---------------------------------------------------
    calib_json = Path("captures/cam_calib.json")
    input_image = Path(r"captures/calibrationSnapshots/charuco_0222.png")
    output_image = input_image.with_name(input_image.stem + "_undist_compare.png")

    # --- load calibration ----------------------------------------------------
    with open(calib_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    K = np.array(data["intrinsic_matrix"], dtype=np.float32)
    dist = np.array(data["distortion_coeffs"], dtype=np.float32).reshape(-1, 1)
    w0 = int(data["image_width"])
    h0 = int(data["image_height"])

    print(f"[INFO] Calibration size: {w0} x {h0}")
    print(f"[INFO] K =\n{K}")
    print(f"[INFO] dist = {dist.ravel()}")

    # --- load image ---------------------------------------------------------
    img = cv2.imread(str(input_image), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"[ERROR] Could not read image: {input_image}")

    h, w = img.shape[:2]
    print(f"[INFO] Input image size: {w} x {h}")

    if (w, h) != (w0, h0):
        print("[WARN] Image size != calibration size – undistortion may be off.")

    # --- plain undistort (no ROI tricks) ------------------------------------
    undist = cv2.undistort(img, K, dist)

    # --- build side-by-side comparison --------------------------------------
    # Resize to same height just in case (should already match)


    compare = np.hstack([img, undist])
    cv2.drawChessboardCorners
    cv2.imwrite(str(output_image), compare)
    print(f"[INFO] Saved side-by-side comparison to: {output_image}")

    # --- show on screen ------------------------------------------------------
    cv2.imshow("Original (left) | Undistorted (right)", compare)
    print("[INFO] Press any key in the image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
