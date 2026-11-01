"""Frozen acoustic domain mixture; no online fitting or filename/codec class rules."""
import math
import numpy as np


def logsumexp(values):
    high = float(np.max(values))
    return high + math.log(float(np.exp(values - high).sum()))


class DomainGate:
    def __init__(self, config):
        if config['classes'] != ['CFAD', 'InTheWild', 'LlamaPartialSpoof', 'ODSS']:
            raise ValueError('Unknown expert domains')
        self.weights = np.asarray(config['linear_weights'], dtype=np.float64)
        self.bias = np.asarray(config['class_bias'], dtype=np.float64)
        self.calibration = np.asarray(config['expert_calibrations'], dtype=np.float64)
        self.temperature = float(config['temperature'])
        self.divisor = float(config['score_divisor'])
        if (self.weights.shape != (4, 1024) or self.bias.shape != (4,) or
                self.calibration.shape != (4, 2) or not (self.calibration[:, 0] > 0).all() or
                not all(np.isfinite(x).all() for x in [self.weights, self.bias, self.calibration]) or
                not math.isfinite(self.temperature) or self.temperature < 0 or
                not math.isfinite(self.divisor) or self.divisor <= 0):
            raise ValueError('Invalid frozen domain parameters')

    def score(self, embedding, short_tta, v2):
        x = np.asarray(embedding, dtype=np.float64)
        if x.shape != (1024,) or not np.isfinite(x).all() or not all(math.isfinite(v) for v in [short_tta, v2]):
            raise ValueError('Invalid domain mixture input')
        logits = (self.weights @ x + self.bias) / self.divisor * self.temperature
        if not np.isfinite(logits).all():
            raise ValueError('Non-finite domain logits')
        log_domain = logits - logsumexp(logits)
        expert = np.asarray([v2, v2, short_tta, short_tta]) * self.calibration[:, 0] + self.calibration[:, 1]
        fake = logsumexp(log_domain - np.logaddexp(0., -expert))
        real = logsumexp(log_domain - np.logaddexp(0., expert))
        score = fake - real
        if not math.isfinite(score):
            raise ValueError('Non-finite mixture score')
        probabilities = np.exp(log_domain)
        return {'score': score, 'domain_probabilities': probabilities.tolist(),
                'expert_logits': expert.tolist(), 'p_v2': float(probabilities[:2].sum())}
