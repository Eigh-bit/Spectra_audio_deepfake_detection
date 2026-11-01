"""Explicit output policy; user-provided semantics are distinct from official verification."""
import math


def sigmoid_float64(value):
    """Stable binary64 sigmoid; preserve tiny outputs and near-one representable values."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Non-finite sigmoid input')
    if value >= 0.0:
        exponential = math.exp(-value)
        # Avoid rounding 1+exp(-x) to 1 prematurely (e.g. x=37).
        return 1.0 - exponential / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


class OutputAdapter:
    def __init__(self, config, require_official=False):
        self.config = dict(config)
        self.direction = config['confidence_direction']
        self.transform = config['confidence_transform']
        self.threshold = float(config['fake_threshold'])
        self.labels = config['labels']
        if self.direction not in ('fake_high', 'real_high'):
            raise ValueError('Explicit confidence direction required')
        if self.transform not in ('raw', 'sigmoid') or not math.isfinite(self.threshold):
            raise ValueError('Invalid transform or threshold')
        if set(self.labels) != {'real', 'fake'} or self.labels['real'] == self.labels['fake']:
            raise ValueError('Distinct real/fake label values required')
        if any(str(v) not in ('real', 'fake', '0', '1') for v in self.labels.values()):
            raise ValueError('Label value absent from official example')
        if require_official and not (config.get('official_verified') and config.get('evidence')):
            raise ValueError('Official output mapping needs verified audio-task evidence; local mode remains available')

    def confidence(self, fake_score):
        score = float(fake_score)
        if not math.isfinite(score):
            raise ValueError('Non-finite fake score; no fallback prediction')
        oriented = score if self.direction == 'fake_high' else -score
        return oriented if self.transform == 'raw' else sigmoid_float64(oriented)

    def label_from_confidence(self, confidence):
        confidence = float(confidence)
        if not math.isfinite(confidence) or (self.transform == 'sigmoid' and not 0 <= confidence <= 1):
            raise ValueError('Invalid confidence for label decision')
        boundary = self.confidence(self.threshold)
        # Keep the configured native threshold, including its fake-on-tie convention.
        # Decide on representable confidence to keep CSV labels consistent near 0.5.
        fake = confidence >= boundary if self.direction == 'fake_high' else confidence <= boundary
        return self.labels['fake' if fake else 'real']

    def prediction(self, fake_score):
        confidence = self.confidence(fake_score)
        return self.label_from_confidence(confidence), confidence
