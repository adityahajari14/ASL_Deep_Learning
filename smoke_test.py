"""Run once after installation: python smoke_test.py (no training or camera)."""
from pathlib import Path
from model import SignModel
from recognition import Recognizer, Settings
import numpy as np
ROOT = Path(__file__).parent
model = SignModel(ROOT / 'model_results')
assert model.network.count_params() == 1016201
scores = model.predict(np.zeros((160,160,3), dtype=np.uint8))
assert scores.shape == (25,)
recognizer = Recognizer(model, ROOT / 'hand_landmarker.task')
try:
    _, result = recognizer.process(np.zeros((480,640,3), dtype=np.uint8), Settings())
    assert result['label'] is None
    assert result['crop'] is None
finally:
    recognizer.close()
print('PASS: model load, 25 scores, parameter count and no-hand detection.')
