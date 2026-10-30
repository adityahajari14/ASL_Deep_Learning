"""Keep each visitor's hand tracker and voting history separate.

Video frames are returned to the browser straight away. Recognition runs in a
background thread on the newest frame only, so a slow server lowers the
prediction rate instead of making the camera preview lag.
"""
from threading import Event, Lock, Thread
from time import monotonic, sleep
import logging
import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase
from recognition import Recognizer, Settings

MAX_PREDICTIONS_PER_SECOND = 8
BOX_COLOUR = (20, 150, 125)


class VideoProcessor(VideoProcessorBase):
    def __init__(self, model, detector_path):
        self.lock = Lock()
        self.recognizer = Recognizer(model, detector_path)
        self.settings = Settings()
        self.result = None
        self.bounds = None
        self.latest = None
        self.stopped = False
        self.has_frame = Event()
        self.worker = Thread(target=self._recognize_loop, daemon=True)
        self.worker.start()

    def configure(self, settings):
        with self.lock:
            self.settings = settings

    def snapshot(self):
        with self.lock:
            return self.result

    def recv(self, frame):
        rgb = frame.to_ndarray(format='rgb24')
        with self.lock:
            self.latest = rgb
            flip, bounds = self.settings.flip_input, self.bounds
        self.has_frame.set()
        if flip:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
        elif not bounds:
            return frame
        if bounds:
            x0, y0, x1, y1 = bounds
            rgb = rgb.copy()
            cv2.rectangle(rgb, (x0, y0), (x1 - 1, y1 - 1), BOX_COLOUR, 2)
        return av.VideoFrame.from_ndarray(rgb, format='rgb24')

    def _recognize_loop(self):
        try:
            while True:
                self.has_frame.wait()
                self.has_frame.clear()
                with self.lock:
                    if self.stopped:
                        return
                    rgb, settings = self.latest, self.settings
                started = monotonic()
                try:
                    _, result = self.recognizer.process(rgb, settings)
                    bounds = result['bounds']
                except Exception:
                    logging.exception('Video recognition failed')
                    self.recognizer.close()  # A fresh detector is created on the next frame.
                    result, bounds = {'error': 'Recognition failed. Retrying…'}, None
                with self.lock:
                    self.result, self.bounds = result, bounds
                sleep(max(0.0, 1 / MAX_PREDICTIONS_PER_SECOND - (monotonic() - started)))
        finally:
            self.recognizer.close()

    def on_ended(self):
        with self.lock:
            self.stopped = True
            self.result = None
            self.bounds = None
        self.has_frame.set()
