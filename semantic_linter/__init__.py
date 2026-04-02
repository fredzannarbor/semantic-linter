"""
semantic-linter — SAX-based narrative pacing quality control.

Computes inter-paragraph novelty curves from text embeddings,
applies Symbolic Aggregate approXimation (SAX), and compares
motif distributions against genre-specific target profiles.

Usage:
    from semantic_linter import PacingLinter

    linter = PacingLinter()
    report = linter.lint_text(text, profile="risk_atlas")
    print(report.summary())
"""

from .linter import (
    PacingLinter,
    PacingProfile,
    PacingReport,
    PacingFlag,
    Severity,
    ScalarDynamics,
    PROFILES,
    paa,
    z_normalize,
    sax_discretize,
    sax_transform,
    extract_motifs,
    split_paragraphs,
    compute_novelty_curve,
)
from .annotator import PacingAnnotator, annotate_from_report

__version__ = "0.1.0"
__all__ = [
    "PacingLinter",
    "PacingProfile",
    "PacingReport",
    "PacingFlag",
    "Severity",
    "ScalarDynamics",
    "PROFILES",
    "PacingAnnotator",
    "annotate_from_report",
    "paa",
    "z_normalize",
    "sax_discretize",
    "sax_transform",
    "extract_motifs",
    "split_paragraphs",
    "compute_novelty_curve",
]
