"""Keep each visitor's hand tracker and voting history separate."""
from threading import Lock
import logging
import av
from streamlit_webrtc import VideoProcessorBase
from recognition import Recognizer, Settings


class VideoProcessor(VideoProcessorBase):
    def __init__(self, model, detector_path):
        self.lock = Lock()
        self.recognizer = Recognizer(model, detector_path)
        self.settings = Settings()
        self.result = None

    def configure(self, settings):
        with self.lock:
            self.settings = settings

    def snapshot(self):
        with self.lock:
            return self.result

    def recv(self, frame):
        rgb = frame.to_ndarray(format='rgb24')
        with self.lock:
            try:
                image, self.result = self.recognizer.process(rgb, self.settings)
            except Exception:
                logging.exception('Video recognition failed')
                self.recognizer.close()
                self.result = {'error': 'Recognition stopped. Stop and restart the camera.'}
                image = rgb
        return av.VideoFrame.from_ndarray(image, format='rgb24')

    def on_ended(self):
        with self.lock:
            self.recognizer.close()
            self.result = None
