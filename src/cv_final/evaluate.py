"""Binary segmentation metrics restricted to explicitly valid pixels."""
import numpy as np


def evaluate_segmentation(pred, ground_truth, valid_mask):
    """Return confusion counts and finite scores on nonempty valid support.

    All inputs must be equal-shape 2D boolean arrays. Pixels outside support
    are ignored (including ignored ground-truth labels). Empty prediction
    and truth score 1 for every metric; otherwise undefined precision/recall
    score 0. Counts are Python ints and scores are Python floats.
    """
    arrays = (pred, ground_truth, valid_mask)
    if any(not isinstance(a, np.ndarray) or a.dtype != np.bool_ or a.ndim != 2
           for a in arrays):
        raise ValueError("pred, ground_truth and valid_mask must be 2D boolean arrays")
    if any(a.shape != pred.shape for a in arrays):
        raise ValueError("pred, ground_truth and valid_mask must have matching shapes")
    if not np.any(valid_mask):
        raise ValueError("valid_mask must contain at least one valid pixel")
    p, g = pred[valid_mask], ground_truth[valid_mask]
    tp = int(np.count_nonzero(p & g))
    fp = int(np.count_nonzero(p & ~g))
    fn = int(np.count_nonzero(~p & g))
    tn = int(np.count_nonzero(~p & ~g))
    union = tp + fp + fn
    if union == 0:
        return dict(tp=tp, fp=fp, fn=fn, tn=tn, iou=1.0,
                    precision=1.0, recall=1.0, f1=1.0, valid_pixels=int(p.size))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, iou=tp / union,
                precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else 0.0,
                f1=2 * tp / (2 * tp + fp + fn), valid_pixels=int(p.size))
