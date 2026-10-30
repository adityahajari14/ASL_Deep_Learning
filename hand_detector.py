"""Find one hand with MediaPipe Tasks and crop its image."""
from pathlib import Path
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

HAND_MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/'
                  'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task')


def get_hand_model(path):
    path = Path(path)
    if not path.is_file():
        print('Downloading the MediaPipe hand detector (first run only)...')
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix('.part')
        try:
            with urllib.request.urlopen(HAND_MODEL_URL, timeout=120) as source:
                with partial.open('wb') as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
            partial.replace(path)
        except Exception as exc:
            partial.unlink(missing_ok=True)
            raise RuntimeError('Could not download the hand detector. Check your Internet '
                               'connection and run again. See README.md for manual download.') from exc
    return path


def make_detector(path):
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(get_hand_model(path).resolve())),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return vision.HandLandmarker.create_from_options(options)


def hand_crop(rgb, landmarks, margin=0.2):
    height, width = rgb.shape[:2]
    xs = np.array([point.x * width for point in landmarks])
    ys = np.array([point.y * height for point in landmarks])
    side = max(2, int(max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + 2 * margin)))
    left = int((xs.min() + xs.max() - side) / 2)
    top = int((ys.min() + ys.max() - side) / 2)
    right, bottom = left + side, top + side
    x0, y0, x1, y1 = max(0, left), max(0, top), min(width, right), min(height, bottom)
    if x1 <= x0 or y1 <= y0:
        return None, None
    crop = np.zeros((side, side, 3), dtype=np.uint8)
    crop[y0-top:y1-top, x0-left:x1-left] = rgb[y0:y1, x0:x1]
    return crop, (x0, y0, x1, y1)


def context_crop(rgb, landmarks, scale=1.8, wrist_shift=0.15):
    """Include real background and more wrist, without rotating the sign."""
    height, width = rgb.shape[:2]
    if not 1.0 <= scale <= 4.0 or not 0 <= wrist_shift <= 0.5:
        raise ValueError('Use crop scale 1..4 and wrist shift 0..0.5.')
    if height < 2 or width < 2:
        raise ValueError('Camera image is too small.')
    points = np.array([(point.x * width, point.y * height) for point in landmarks])
    if points.shape != (21, 2) or not np.isfinite(points).all():
        return None, None
    low, high = points.min(axis=0), points.max(axis=0)
    span = max(float((high-low).max()), 2.0)
    center = (low+high)/2
    # Landmark 0 is the wrist; 9 is the middle-finger knuckle. Their direction
    # points toward the forearm, including for sideways signs such as G and H.
    direction = points[0] - points[9]
    length = np.linalg.norm(direction)
    if length > 1:
        center += wrist_shift * span * direction / length
    # Keep the full hand inside the square even after shifting toward the wrist.
    required = 2 * np.abs(points-center).max()
    side = min(min(height, width), max(2, int(np.ceil(max(span*scale, required)))))
    left = min(max(int(center[0]-side/2), 0), width-side)
    top = min(max(int(center[1]-side/2), 0), height-side)
    # If the camera limits the square size, favour retaining visible fingers
    # over the requested wrist shift. A hand wider than the square must move back.
    min_left, max_left = max(0, int(np.ceil(high[0]))-side), min(width-side, int(np.floor(low[0])))
    min_top, max_top = max(0, int(np.ceil(high[1]))-side), min(height-side, int(np.floor(low[1])))
    if min_left <= max_left:
        left = min(max(left, min_left), max_left)
    if min_top <= max_top:
        top = min(max(top, min_top), max_top)
    right, bottom = left+side, top+side
    return rgb[top:bottom, left:right].copy(), (left, top, right, bottom)
