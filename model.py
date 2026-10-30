"""Load the existing classifier. This file performs inference, not training."""
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')

import json
from pathlib import Path
from threading import Lock
import cv2
import keras
import numpy as np


class SignModel:
    def __init__(self, folder):
        folder = Path(folder)
        self.config = json.loads((folder / 'inference_config.json').read_text())
        self.names = self.config['classes']
        self.size = self.config['image_size']
        self.lock = Lock()
        self.network = keras.models.load_model(folder / 'best_model.keras', compile=False)
        if self.network.input_shape[1:] != (self.size, self.size, 3):
            raise ValueError('Unexpected model input shape.')
        if self.network.output_shape[-1] != len(self.names):
            raise ValueError('Model labels do not match the output layer.')

    def prepare_image(self, crop):
        if crop.ndim != 3 or crop.shape[2] != 3 or crop.shape[0] != crop.shape[1]:
            raise ValueError('Expected a square RGB image.')
        # Match the notebook: resize float pixels, truncate to uint8, then infer.
        image = cv2.resize(crop.astype(np.float32), (self.size, self.size),
                           interpolation=cv2.INTER_LINEAR)
        return np.clip(image, 0, 255).astype(np.uint8)

    def predict(self, crop):
        image = self.prepare_image(crop)
        # Multiple visitors share the weights, but never call them concurrently.
        with self.lock:
            scores = self.network(image[None].astype(np.float32), training=False).numpy()[0]
        if not np.isfinite(scores).all() or not np.isclose(scores.sum(), 1, atol=1e-4):
            raise RuntimeError('Invalid classifier output.')
        return scores
