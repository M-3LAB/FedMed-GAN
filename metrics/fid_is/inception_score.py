import numpy as np

__all__ = ['calculate_is']

def calculate_is(probs, splits=10):
    # Inception Score
    scores = []
    for i in range(splits):
        part = probs[
            (i * probs.shape[0] // splits):
            ((i + 1) * probs.shape[0] // splits), :]

        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))

    inception_score, std = (np.mean(scores), np.std(scores))

    del probs, scores
    return inception_score, std
