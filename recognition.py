"""One recognition pipeline for both live video and photo testing."""
from dataclasses import dataclass
from time import monotonic, monotonic_ns
import cv2
import mediapipe as mp
import numpy as np
from hand_detector import context_crop, make_detector
from decision import decide, TemporalVote


@dataclass(frozen=True)
class Settings:
    crop_mode: str = 'Automatic hand crop'
    crop_scale: float = 1.8
    flip_input: bool = False
    threshold: float = 0.75
    minimum_gap: float = 0.15


class Recognizer:
    def __init__(self, model, detector_path):
        self.model = model
        self.detector_path = detector_path
        self.detector = None
        self.vote = TemporalVote()
        self.timestamp = 0
        self.previous_settings = None

    def close(self):
        if self.detector is not None:
            self.detector.close()
            self.detector = None
        self.vote.clear()

    def process(self, rgb, settings, use_voting=True):
        if settings != self.previous_settings:
            self.vote.clear()
            self.previous_settings = settings
        # This is explicit because mirror preference must not silently alter signs.
        if settings.flip_input:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
        height, width = rgb.shape[:2]
        crop = None
        bounds = None
        reason = 'No hand - show your hand and wrist'
        if settings.crop_mode == 'Fixed guide box':
            side = int(min(height, width) * 0.80)
            x, y = (width-side)//2, (height-side)//2
            bounds = (x, y, x+side, y+side)
            crop = rgb[y:y+side, x:x+side].copy()
        else:
            if self.detector is None:
                self.detector = make_detector(self.detector_path)
            self.timestamp = max(self.timestamp+1, monotonic_ns()//1_000_000)
            result = self.detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self.timestamp)
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                points = np.array([(p.x*width, p.y*height) for p in landmarks])
                span = np.ptp(points, axis=0).max()
                if (points[:, 0].min() < 3 or points[:, 1].min() < 3 or
                    points[:, 0].max() > width-3 or points[:, 1].max() > height-3):
                    reason = 'Move back - keep every finger inside the frame'
                elif span < 45:
                    reason = 'Move closer - the hand is too small'
                else:
                    crop, bounds = context_crop(rgb, landmarks, settings.crop_scale)
        top = []
        label = None
        preview = None
        if crop is not None:
            preview = self.model.prepare_image(crop)
            label, reason, top = decide(self.model.predict(crop), self.model.names,
                                       settings.threshold, settings.minimum_gap)
        stable = self.vote.update(label, monotonic()) if use_voting else label
        if stable:
            reason = 'Recognized' if use_voting else 'Single photo prediction'
        result = {'label': stable, 'status': reason, 'top': top, 'crop': preview,
                  'settings': settings, 'bounds': bounds, 'time': monotonic()}
        display = rgb.copy()
        if bounds:
            x0, y0, x1, y1 = bounds
            cv2.rectangle(display, (x0, y0), (x1-1, y1-1), (20, 150, 125), 2)
        return display, result
