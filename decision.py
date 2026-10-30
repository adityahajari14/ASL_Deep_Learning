"""Reject ambiguous predictions and smooth accepted ones over time."""
from collections import Counter, deque
import numpy as np


def decide(scores, names, threshold=0.75, minimum_gap=0.15):
    order = np.argsort(scores)[::-1]
    first, second = int(order[0]), int(order[1])
    label = names[first]
    score = float(scores[first])
    gap = score - float(scores[second])
    top = [(names[int(i)], float(scores[i])) for i in order[:3]]
    if label == 'nothing':
        return None, 'No sign', top
    if score < threshold or gap < minimum_gap:
        return None, 'Uncertain - adjust your hand', top
    return label, 'Hold the sign', top


class TemporalVote:
    def __init__(self, window=7, required=5):
        self.history = deque(maxlen=window)
        self.required = required
        self.last_time = None

    def clear(self):
        self.history.clear()
        self.last_time = None

    def update(self, label, now):
        # Old frames must not survive a pause or an uncertain/no-hand frame.
        if self.last_time is not None and now - self.last_time > 1.0:
            self.clear()
        if label is None:
            self.clear()
            return None
        self.last_time = now
        self.history.append(label)
        winner, count = Counter(self.history).most_common(1)[0]
        # Do not keep displaying the previous letter after the sign changes.
        return winner if winner == label and count >= self.required else None
