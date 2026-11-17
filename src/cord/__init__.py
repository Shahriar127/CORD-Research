"""CORD: conformal evidence fusion for backdoor poison detection.

The package is layered so that the statistical core never depends on deep learning code:

* :mod:`cord.stats`     conformal p-/e-values, BH / e-BH, exact presence test, detection floors (NumPy only)
* :mod:`cord.data`      dataset loaders (array based, size agnostic)
* :mod:`cord.attacks`   poisoning attacks applied to the untrusted pool
* :mod:`cord.splits`    random hold-out (C_tr, C_cal) and cross-fitting folds
* :mod:`cord.models`    classifiers exposing intermediate representations
* :mod:`cord.training`  GPU-resident cross-fitted training with checkpoint trajectories
* :mod:`cord.features`  one-pass feature cache per fold, frozen self-supervised encoders
* :mod:`cord.signals`   detection signals computed from the cache (assumption A2 by construction)
* :mod:`cord.fusion`    positive-unlabeled fusion (AdaDetect-style) and naive fusion baselines
* :mod:`cord.pipeline`  calibration, audit and the three outputs (presence, removal set, proxy set)
* :mod:`cord.runner`    resumable stage orchestration driven by a validated config
"""

__version__ = "0.1.0"
