"""rubricon-field: the Rubricon evaluation harness applied to real human annotations.

Three studies over the HateXplain corpus (Mathew et al., AAAI 2021):

``study_a``
    Label-quality audit: agreement battery, consensus structure, the binary
    contrast, annotator load and exposure, per-community agreement, the accuracy
    ceiling, and the signal gate over claims a paper might make.
``study_b``
    Judge-validation harness, exercised end to end with a held-out human in the
    judge slot. No model is called anywhere in this package.
``study_c``
    Replication economics: what raters and items are worth at this dataset's
    measured variance.

Every statistic is computed by ``rubricon``; nothing numerical is reimplemented
here. This package supplies the loader, the study designs, the interpretations,
and the honesty about what each number can and cannot support.
"""

from __future__ import annotations

__version__ = "1.0.0"

from .data import Dataset, load_hatexplain, majority_label, reliability_matrix
from .study_a import run_study_a
from .study_b import run_study_b
from .study_c import run_study_c

__all__ = [
    "Dataset",
    "__version__",
    "load_hatexplain",
    "majority_label",
    "reliability_matrix",
    "run_study_a",
    "run_study_b",
    "run_study_c",
]
