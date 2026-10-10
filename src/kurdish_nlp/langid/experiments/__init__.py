"""Reproducible supervised language-identification experiments.

The modules in this package train, select, calibrate and evaluate baseline
classifiers on a frozen, hash-verified dataset release. Importing the package
does not import numpy, scikit-learn or fastText; each backend module imports its
optional dependencies itself. Every experiment parameter is read from a versioned
JSON configuration (``configs/langid/experiments/*.json``).

Protocol, enforced by :mod:`kurdish_nlp.langid.experiments.runner`:

* ``train`` fits model parameters only;
* ``dev`` is split into group-disjoint ``dev_tune`` (hyperparameter selection)
  and ``dev_calib`` (temperature scaling and abstention thresholds);
* the internal ``test`` split and the external UD benchmark are evaluated only
  after a selection lock has frozen models, calibration and thresholds.
"""
