"""
Pacing Linter — SAX-based narrative quality control layer.

Computes novelty curves from paragraph embeddings, applies SAX discretization,
and compares the resulting motif distribution against genre-specific target
profiles. Produces actionable revision diagnostics.

Usage:
    from codexes.modules.rkhs.pacing_linter import PacingLinter, PacingProfile

    linter = PacingLinter()
    report = linter.lint_text(text, profile="middle_grade_fiction")
    print(report.summary())
    for flag in report.flags:
        print(flag)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from enum import Enum
import json
import re


# ---------------------------------------------------------------------------
# SAX Implementation
# ---------------------------------------------------------------------------

def paa(curve: np.ndarray, n_segments: int) -> np.ndarray:
    """Piecewise Aggregate Approximation: reduce curve to n_segments means."""
    seg_len = len(curve) / n_segments
    return np.array([
        np.mean(curve[int(i * seg_len):int((i + 1) * seg_len)])
        for i in range(n_segments)
    ])


def z_normalize(values: np.ndarray) -> np.ndarray:
    """Z-normalize to zero mean, unit variance."""
    std = np.std(values)
    if std < 1e-10:
        return np.zeros_like(values)
    return (values - np.mean(values)) / std


# Gaussian quantile breakpoints for SAX alphabets
SAX_BREAKPOINTS = {
    3: np.array([-0.43, 0.43]),
    4: np.array([-0.67, 0.0, 0.67]),
    5: np.array([-0.84, -0.25, 0.25, 0.84]),
    7: np.array([-1.07, -0.57, -0.18, 0.18, 0.57, 1.07]),
}


def sax_discretize(z_values: np.ndarray, alphabet_size: int = 5) -> str:
    """Map z-normalized values to SAX string."""
    breaks = SAX_BREAKPOINTS[alphabet_size]
    chars = 'abcdefg'[:alphabet_size]
    result = []
    for v in z_values:
        idx = np.searchsorted(breaks, v)
        result.append(chars[idx])
    return ''.join(result)


def extract_motifs(sax_string: str, k: int = 4) -> Dict[str, int]:
    """Extract overlapping k-gram motif frequencies."""
    counts = {}
    for i in range(len(sax_string) - k + 1):
        motif = sax_string[i:i + k]
        counts[motif] = counts.get(motif, 0) + 1
    return counts


def sax_transform(curve: np.ndarray, n_segments: int = 16,
                  alphabet_size: int = 5, k_gram: int = 4
                  ) -> Tuple[str, Dict[str, int]]:
    """Full SAX pipeline: curve → PAA → z-norm → discretize → motifs."""
    paa_vals = paa(curve, n_segments)
    z_vals = z_normalize(paa_vals)
    sax_str = sax_discretize(z_vals, alphabet_size)
    motifs = extract_motifs(sax_str, k_gram)
    return sax_str, motifs


# ---------------------------------------------------------------------------
# Novelty Curve Computation
# ---------------------------------------------------------------------------

def compute_novelty_curve(paragraphs: List[str], embedder=None) -> np.ndarray:
    """
    Compute inter-paragraph cosine novelty curve.

    Returns array of length len(paragraphs)-1 where each value is
    1 - cos_sim(paragraph_i, paragraph_{i+1}).
    """
    if embedder is None:
        embedder = _get_default_embedder()

    embeddings = embedder.embed(paragraphs)

    novelty = []
    for i in range(len(embeddings) - 1):
        a, b = embeddings[i], embeddings[i + 1]
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        novelty.append(1.0 - cos_sim)
    return np.array(novelty)


def _get_default_embedder():
    """Get the best available embedder. Requires sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "semantic-linter requires sentence-transformers. "
            "Install with: pip install sentence-transformers"
        )

    class _STEmbedder:
        def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
            self.model = SentenceTransformer(model_name, trust_remote_code=True)

        def embed(self, texts: List[str]) -> np.ndarray:
            return self.model.encode(texts, show_progress_bar=False)

    return _STEmbedder()


def split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs, filtering empties and structural elements."""
    paras = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paras if _is_substantive(p.strip())]


def _is_substantive(text: str) -> bool:
    """Return True if text is a substantive content paragraph, not a heading or label.

    Filters out:
    - Empty/short strings (< 30 chars)
    - Numeric-only lines (chapter numbers)
    - Markdown headings (# Title)
    - Lines with no sentence-ending punctuation and < 100 chars (likely titles/headers)
    - Lines that are ALL CAPS or Title Case with no sentence structure
    """
    if len(text) < 30:
        return False
    # Pure numbers (chapter numbers like "18")
    if re.match(r'^\d+\.?\s*$', text):
        return False
    # Markdown headings
    if re.match(r'^#{1,6}\s', text):
        return False
    # Short lines with no sentence-ending punctuation — likely titles/headers
    if len(text) < 100 and not re.search(r'[.!?:;]\s*$', text):
        return False
    return True


# ---------------------------------------------------------------------------
# Scalar Dynamics
# ---------------------------------------------------------------------------

@dataclass
class ScalarDynamics:
    """Book-level scalar features of a novelty curve."""
    mean_novelty: float
    std_novelty: float
    speed: float  # mean |delta|
    volume: float  # sum |delta|
    circuitousness: float  # volume / |end - start|
    reversal_count: int  # direction changes
    ti_ratio: float  # trend-to-irregularity ratio

    @classmethod
    def from_curve(cls, curve: np.ndarray) -> 'ScalarDynamics':
        deltas = np.diff(curve)
        abs_deltas = np.abs(deltas)
        net = abs(curve[-1] - curve[0]) if len(curve) > 1 else 1e-10

        # Count reversals (sign changes in deltas)
        signs = np.sign(deltas)
        reversals = int(np.sum(signs[:-1] != signs[1:]))

        # Trend-to-irregularity
        half = len(curve) // 2
        trend = abs(np.mean(curve[half:]) - np.mean(curve[:half]))
        std = np.std(curve)
        ti = trend / std if std > 1e-10 else 0.0

        return cls(
            mean_novelty=float(np.mean(curve)),
            std_novelty=float(np.std(curve)),
            speed=float(np.mean(abs_deltas)) if len(abs_deltas) > 0 else 0.0,
            volume=float(np.sum(abs_deltas)),
            circuitousness=float(np.sum(abs_deltas) / max(net, 1e-10)),
            reversal_count=reversals,
            ti_ratio=float(ti),
        )


# ---------------------------------------------------------------------------
# Pacing Profiles — Target SAX patterns per genre
# ---------------------------------------------------------------------------

@dataclass
class PacingProfile:
    """Target pacing specification for a genre/audience."""
    name: str
    description: str

    # Scalar targets (acceptable ranges)
    mean_novelty_range: Tuple[float, float] = (0.03, 0.08)
    std_novelty_range: Tuple[float, float] = (0.005, 0.03)
    circuitousness_range: Tuple[float, float] = (2.0, 20.0)

    # SAX parameters
    paa_segments: int = 16
    alphabet_size: int = 5
    k_gram: int = 4

    # Desired motifs (high-frequency = good pacing)
    preferred_motifs: List[str] = field(default_factory=list)
    # Undesired motifs (flag if frequent)
    avoided_motifs: List[str] = field(default_factory=list)

    # Chapter-level constraints
    max_flat_run: int = 4  # max consecutive same-letter SAX symbols
    min_chapter_opening_novelty: str = 'c'  # minimum SAX level for chapter starts
    preferred_chapter_ending_novelty: str = 'd'  # SAX level for chapter ends

    # Window analysis
    window_size: int = 20  # paragraphs per window
    window_paa: int = 8

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d

    def save(self, path: Path):
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> 'PacingProfile':
        return cls(**json.loads(path.read_text()))


# Pre-defined profiles
PROFILES: Dict[str, PacingProfile] = {
    # NOTE: Nomic Embed Text v1.5 inter-paragraph cosine distances typically
    # range 0.25-0.45 for prose. Profiles are calibrated to this scale.
    # Other embedding models will require recalibration.
    "middle_grade_fiction": PacingProfile(
        name="middle_grade_fiction",
        description="Grade 5-8 fiction. High engagement, no dead zones, "
                    "double-bump chapter rhythm, cliffhanger endings.",
        mean_novelty_range=(0.28, 0.40),
        std_novelty_range=(0.06, 0.12),
        circuitousness_range=(10.0, 200.0),  # should go somewhere
        preferred_motifs=["bcdc", "cdcb", "bcdb", "cdcd", "dcdc"],
        avoided_motifs=["aaaa", "aabb", "bbbb", "abab", "aaba"],
        max_flat_run=3,
        min_chapter_opening_novelty='c',
        preferred_chapter_ending_novelty='d',
        window_size=20,
        window_paa=8,
    ),
    "risk_atlas": PacingProfile(
        name="risk_atlas",
        description="Risk monitoring atlas (Food Shock type). High inter-chapter "
                    "novelty, briefing structure within chapters, high circuitousness.",
        mean_novelty_range=(0.28, 0.40),
        std_novelty_range=(0.07, 0.12),
        circuitousness_range=(25.0, 800.0),  # circling back is good
        preferred_motifs=["bccb", "bccd", "ccbb", "cbbc"],
        avoided_motifs=["edcb", "dcba", "edca", "aaaa", "bbbb"],
        max_flat_run=4,
        min_chapter_opening_novelty='c',
        preferred_chapter_ending_novelty='b',
        window_size=40,
        window_paa=8,
    ),
    "literary_fiction": PacingProfile(
        name="literary_fiction",
        description="Adult literary fiction. Longer wavelength oscillation, "
                    "tolerance for slow sections, resolution at chapter end.",
        mean_novelty_range=(0.25, 0.40),
        std_novelty_range=(0.06, 0.14),
        circuitousness_range=(15.0, 500.0),
        preferred_motifs=["bcde", "cdec", "dedc", "bccd"],
        avoided_motifs=["aaaa", "bbbb", "abab"],
        max_flat_run=5,
        min_chapter_opening_novelty='b',
        preferred_chapter_ending_novelty='b',
        window_size=40,
        window_paa=8,
    ),
}


# ---------------------------------------------------------------------------
# Pacing Flags
# ---------------------------------------------------------------------------

class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class PacingFlag:
    """A single pacing diagnostic."""
    severity: Severity
    location: str  # e.g., "Chapter 3, paragraphs 15-25"
    issue: str
    suggestion: str
    metric_name: str = ""
    metric_value: float = 0.0
    metric_target: str = ""

    def __str__(self):
        icon = {"info": ".", "warning": "!", "error": "X"}[self.severity.value]
        return f"[{icon}] {self.location}: {self.issue}\n    → {self.suggestion}"


# ---------------------------------------------------------------------------
# Pacing Report
# ---------------------------------------------------------------------------

@dataclass
class PacingReport:
    """Complete pacing analysis results."""
    profile_name: str
    novelty_curve: np.ndarray
    sax_string: str
    motifs: Dict[str, int]
    scalars: ScalarDynamics
    flags: List[PacingFlag]

    # Per-chapter analysis (if chapters detected)
    chapter_sax: List[str] = field(default_factory=list)
    chapter_curves: List[np.ndarray] = field(default_factory=list)

    # Window analysis
    window_sax: List[str] = field(default_factory=list)
    window_motifs: List[Dict[str, int]] = field(default_factory=list)

    def summary(self) -> str:
        n_err = sum(1 for f in self.flags if f.severity == Severity.ERROR)
        n_warn = sum(1 for f in self.flags if f.severity == Severity.WARNING)
        n_info = sum(1 for f in self.flags if f.severity == Severity.INFO)

        grade = "PASS" if n_err == 0 else "FAIL"
        if n_err == 0 and n_warn > 0:
            grade = "PASS (with warnings)"

        lines = [
            f"Pacing Lint Report — Profile: {self.profile_name}",
            f"{'=' * 50}",
            f"Grade: {grade}",
            f"Novelty curve: {len(self.novelty_curve)} paragraphs",
            f"SAX string ({len(self.sax_string)} segments): {self.sax_string}",
            f"Scalars: mean={self.scalars.mean_novelty:.4f}, "
            f"speed={self.scalars.speed:.4f}, "
            f"circ={self.scalars.circuitousness:.1f}",
            f"Flags: {n_err} errors, {n_warn} warnings, {n_info} info",
            f"{'=' * 50}",
        ]
        for f in self.flags:
            lines.append(str(f))
        return '\n'.join(lines)

    def to_dict(self) -> dict:
        return {
            "profile": self.profile_name,
            "sax_string": self.sax_string,
            "motifs": self.motifs,
            "scalars": self.scalars.__dict__,
            "n_flags": len(self.flags),
            "flags": [
                {"severity": f.severity.value, "location": f.location,
                 "issue": f.issue, "suggestion": f.suggestion}
                for f in self.flags
            ],
            "chapter_sax": self.chapter_sax,
            "window_sax": self.window_sax,
        }

    def punch_list(self) -> dict:
        """Generate a structured revision punch list consumable by an LLM.

        Returns a JSON-serializable dict with:
        - metadata: profile, grade, sax string, scalar summary
        - revisions: ordered list of actionable items, each with:
            - id: sequential revision ID
            - priority: 1 (must fix), 2 (should fix), 3 (consider)
            - location: where in the text
            - problem: what's wrong (concise, factual)
            - instruction: what to do (imperative, specific)
            - constraint: what NOT to change while fixing this
            - sax_context: the SAX string at this location for reference

        Designed to be passed directly to an LLM as a revision prompt.
        """
        n_err = sum(1 for f in self.flags if f.severity == Severity.ERROR)
        n_warn = sum(1 for f in self.flags if f.severity == Severity.WARNING)
        grade = "FAIL" if n_err > 0 else ("WARN" if n_warn > 0 else "PASS")

        revisions = []
        rev_id = 1

        # Priority 1: Errors (dead zones, structural failures)
        for f in self.flags:
            if f.severity == Severity.ERROR:
                revisions.append(self._flag_to_revision(f, rev_id, priority=1))
                rev_id += 1

        # Priority 2: Warnings (weak openings, flat runs, scalar issues)
        for f in self.flags:
            if f.severity == Severity.WARNING:
                revisions.append(self._flag_to_revision(f, rev_id, priority=2))
                rev_id += 1

        # Priority 3: Info (motif imbalance, style suggestions)
        for f in self.flags:
            if f.severity == Severity.INFO:
                revisions.append(self._flag_to_revision(f, rev_id, priority=3))
                rev_id += 1

        return {
            "punch_list_version": "1.0",
            "profile": self.profile_name,
            "grade": grade,
            "sax_string": self.sax_string,
            "scalars": {
                "mean_novelty": round(self.scalars.mean_novelty, 4),
                "std_novelty": round(self.scalars.std_novelty, 4),
                "circuitousness": round(self.scalars.circuitousness, 1),
                "speed": round(self.scalars.speed, 4),
            },
            "total_revisions": len(revisions),
            "by_priority": {
                "must_fix": sum(1 for r in revisions if r["priority"] == 1),
                "should_fix": sum(1 for r in revisions if r["priority"] == 2),
                "consider": sum(1 for r in revisions if r["priority"] == 3),
            },
            "revisions": revisions,
            "instructions_for_model": (
                "You are revising a manuscript based on pacing analysis. "
                "Each revision item identifies a location, describes the pacing "
                "problem, and gives a specific instruction. Apply revisions in "
                "priority order (1 = must fix, 2 = should fix, 3 = consider). "
                "Preserve the factual content and argument structure — only "
                "change the pacing, transitions, and information sequencing. "
                "After revising, the text should produce a different SAX string "
                "at the flagged locations."
            ),
        }

    def _flag_to_revision(self, flag: PacingFlag, rev_id: int,
                          priority: int) -> dict:
        """Convert a PacingFlag to a structured revision item."""
        # Derive SAX context from chapter_sax if available
        sax_context = ""
        loc = flag.location.lower()
        if "section" in loc:
            import re
            ch_match = re.search(r'section\s+(\d+)', loc)
            if ch_match:
                ch_idx = int(ch_match.group(1)) - 1
                if 0 <= ch_idx < len(self.chapter_sax):
                    sax_context = self.chapter_sax[ch_idx]

        # Generate a constraint based on the problem type
        constraint = "Preserve all factual claims, data points, and citations."
        if "opening" in loc:
            constraint += " Keep the section's core topic — only change the opening hook."
        elif "dead zone" in flag.issue.lower():
            constraint += " Do not add new factual claims — restructure existing content."
        elif "flat run" in flag.issue.lower():
            constraint += " Maintain paragraph count — vary content, not length."

        return {
            "id": f"REV-{rev_id:03d}",
            "priority": priority,
            "priority_label": {1: "must_fix", 2: "should_fix", 3: "consider"}[priority],
            "location": flag.location,
            "problem": flag.issue,
            "instruction": flag.suggestion,
            "constraint": constraint,
            "sax_context": sax_context,
        }

    def save_punch_list(self, path: Path) -> Path:
        """Save punch list as JSON file."""
        data = self.punch_list()
        path.write_text(json.dumps(data, indent=2))
        return path


# ---------------------------------------------------------------------------
# Pacing Linter
# ---------------------------------------------------------------------------

class PacingLinter:
    """
    SAX-based narrative pacing quality control.

    Computes novelty curves, applies SAX discretization, and checks
    against genre-specific target profiles.
    """

    def __init__(self, embedder=None):
        self._embedder = embedder

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = _get_default_embedder()
        return self._embedder

    def lint_text(self, text: str,
                  profile: str = "middle_grade_fiction",
                  chapter_separator: str = r'\n#{1,2}\s',
                  ) -> PacingReport:
        """
        Lint a full text against a pacing profile.

        Args:
            text: Full text (multiple chapters OK)
            profile: Profile name from PROFILES or a PacingProfile instance
            chapter_separator: Regex to split chapters (default: markdown ##)

        Returns:
            PacingReport with diagnostics
        """
        if isinstance(profile, str):
            prof = PROFILES.get(profile)
            if prof is None:
                raise ValueError(f"Unknown profile '{profile}'. "
                                 f"Available: {list(PROFILES.keys())}")
        else:
            prof = profile

        # Split into paragraphs
        paragraphs = split_paragraphs(text)
        if len(paragraphs) < 10:
            return PacingReport(
                profile_name=prof.name,
                novelty_curve=np.array([]),
                sax_string="",
                motifs={},
                scalars=ScalarDynamics(0, 0, 0, 0, 0, 0, 0),
                flags=[PacingFlag(
                    Severity.ERROR, "Whole text",
                    f"Only {len(paragraphs)} paragraphs detected (minimum 10)",
                    "Provide more text for meaningful pacing analysis."
                )],
            )

        # Compute novelty curve
        curve = compute_novelty_curve(paragraphs, self.embedder)

        # SAX transform (whole-book)
        sax_str, motifs = sax_transform(
            curve, prof.paa_segments, prof.alphabet_size, prof.k_gram
        )

        # Scalar dynamics
        scalars = ScalarDynamics.from_curve(curve)

        # Collect flags
        flags = []

        # --- Scalar range checks ---
        self._check_scalar_range(flags, "mean_novelty", scalars.mean_novelty,
                                 prof.mean_novelty_range)
        self._check_scalar_range(flags, "std_novelty", scalars.std_novelty,
                                 prof.std_novelty_range)
        self._check_scalar_range(flags, "circuitousness", scalars.circuitousness,
                                 prof.circuitousness_range)

        # --- Flat run detection ---
        self._check_flat_runs(flags, sax_str, prof.max_flat_run)

        # --- Motif checks ---
        self._check_motifs(flags, motifs, prof)

        # --- Chapter-level analysis ---
        chapter_sax = []
        chapter_curves_list = []
        chapters = re.split(chapter_separator, text)
        chapters = [c.strip() for c in chapters if len(c.strip()) > 200]

        if len(chapters) > 1:
            for i, ch_text in enumerate(chapters):
                ch_paras = split_paragraphs(ch_text)
                if len(ch_paras) < 5:
                    continue
                ch_curve = compute_novelty_curve(ch_paras, self.embedder)
                chapter_curves_list.append(ch_curve)

                ch_sax, _ = sax_transform(
                    ch_curve, min(prof.window_paa, len(ch_curve)),
                    prof.alphabet_size, prof.k_gram
                )
                chapter_sax.append(ch_sax)

                # Chapter heading (first line of the split chunk)
                ch_heading = ch_text.split('\n')[0].strip()[:80]

                # Check chapter opening
                if ch_sax and ch_sax[0] < prof.min_chapter_opening_novelty:
                    flags.append(PacingFlag(
                        Severity.WARNING,
                        f"Section {i + 1} \"{ch_heading}\", opening",
                        f"Opening novelty '{ch_sax[0]}' below target "
                        f"'{prof.min_chapter_opening_novelty}'",
                        "Start with a stronger hook — new setting, "
                        "surprising action, or unanswered question.",
                    ))

                # Check chapter ending
                if ch_sax and ch_sax[-1] < prof.preferred_chapter_ending_novelty:
                    if prof.name == "middle_grade_fiction":
                        flags.append(PacingFlag(
                            Severity.INFO,
                            f"Section {i + 1} \"{ch_heading}\", ending",
                            f"Ending novelty '{ch_sax[-1]}' — consider "
                            f"ending on '{prof.preferred_chapter_ending_novelty}' "
                            f"or higher for a cliffhanger",
                            "End mid-action or with a revelation to pull "
                            "the reader into the next chapter.",
                        ))

        # --- Window-level analysis ---
        window_sax = []
        window_motifs_list = []
        W = prof.window_size
        stride = W // 2
        for start in range(0, len(curve) - W + 1, stride):
            w_curve = curve[start:start + W]
            w_sax, w_motifs = sax_transform(
                w_curve, prof.window_paa, prof.alphabet_size, prof.k_gram
            )
            window_sax.append(w_sax)
            window_motifs_list.append(w_motifs)

            # Check for dead zones (all low novelty)
            if all(c in 'ab' for c in w_sax):
                para_start = start + 1
                para_end = start + W
                flags.append(PacingFlag(
                    Severity.ERROR,
                    f"Paragraphs {para_start}–{para_end}",
                    f"Dead zone detected — SAX '{w_sax}' (all low novelty)",
                    "Inject a pivot, complication, revelation, or "
                    "scene change to break the monotony.",
                ))

            # Check for chaos (all high novelty)
            if all(c in 'de' for c in w_sax):
                para_start = start + 1
                para_end = start + W
                flags.append(PacingFlag(
                    Severity.WARNING,
                    f"Paragraphs {para_start}–{para_end}",
                    f"Sustained high novelty — SAX '{w_sax}' (reader fatigue risk)",
                    "Add a brief consolidation passage — summary, "
                    "reflection, or familiar callback — to let the "
                    "reader absorb before continuing.",
                ))

        return PacingReport(
            profile_name=prof.name,
            novelty_curve=curve,
            sax_string=sax_str,
            motifs=motifs,
            scalars=scalars,
            flags=flags,
            chapter_sax=chapter_sax,
            chapter_curves=chapter_curves_list,
            window_sax=window_sax,
            window_motifs=window_motifs_list,
        )

    def lint_file(self, path: Path,
                  profile: str = "middle_grade_fiction") -> PacingReport:
        """Lint a text file."""
        text = path.read_text(encoding='utf-8', errors='replace')
        return self.lint_text(text, profile=profile)

    # --- Private helpers ---

    @staticmethod
    def _check_scalar_range(flags: List[PacingFlag], name: str,
                            value: float, target_range: Tuple[float, float]):
        lo, hi = target_range
        if value < lo:
            flags.append(PacingFlag(
                Severity.WARNING, "Whole text",
                f"{name} = {value:.4f} is below target range [{lo}, {hi}]",
                f"Increase {name} — the text may feel monotonous or flat.",
                metric_name=name, metric_value=value,
                metric_target=f"[{lo}, {hi}]",
            ))
        elif value > hi:
            flags.append(PacingFlag(
                Severity.WARNING, "Whole text",
                f"{name} = {value:.4f} is above target range [{lo}, {hi}]",
                f"Decrease {name} — the text may feel chaotic or exhausting.",
                metric_name=name, metric_value=value,
                metric_target=f"[{lo}, {hi}]",
            ))

    @staticmethod
    def _check_flat_runs(flags: List[PacingFlag], sax_str: str, max_run: int):
        """Detect runs of repeated SAX symbols."""
        if not sax_str:
            return
        current_char = sax_str[0]
        run_length = 1
        for i in range(1, len(sax_str)):
            if sax_str[i] == current_char:
                run_length += 1
                if run_length > max_run:
                    flags.append(PacingFlag(
                        Severity.WARNING,
                        f"SAX segments {i - run_length + 1}–{i + 1}",
                        f"Flat run of '{current_char}' × {run_length} "
                        f"(max allowed: {max_run})",
                        "Vary the pacing — alternate between exposition "
                        "and action, or shift emotional register.",
                    ))
                    break  # only flag once per run
            else:
                current_char = sax_str[i]
                run_length = 1

    @staticmethod
    def _check_motifs(flags: List[PacingFlag], motifs: Dict[str, int],
                      prof: PacingProfile):
        """Check for preferred/avoided motif patterns."""
        total = sum(motifs.values()) or 1

        # Check for avoided motifs
        for motif in prof.avoided_motifs:
            count = motifs.get(motif, 0)
            freq = count / total
            if freq > 0.1:  # more than 10% of motifs
                flags.append(PacingFlag(
                    Severity.WARNING, "Whole text",
                    f"Avoided motif '{motif}' appears {count} times "
                    f"({freq:.0%} of all motifs)",
                    f"The pattern '{motif}' suggests "
                    + _motif_interpretation(motif),
                ))

        # Check that at least some preferred motifs appear
        pref_count = sum(motifs.get(m, 0) for m in prof.preferred_motifs)
        pref_freq = pref_count / total
        if pref_freq < 0.05 and prof.preferred_motifs:
            flags.append(PacingFlag(
                Severity.INFO, "Whole text",
                f"Preferred motifs ({', '.join(prof.preferred_motifs[:3])}, ...) "
                f"appear rarely ({pref_freq:.0%})",
                "Consider restructuring some passages to follow "
                "the recommended pacing rhythm for this genre.",
            ))


def _motif_interpretation(motif: str) -> str:
    """Human-readable interpretation of a SAX motif."""
    interpretations = {
        "aaaa": "prolonged low novelty — reader may disengage.",
        "bbbb": "sustained below-average novelty — plodding pace.",
        "aabb": "gradual decline into low engagement.",
        "abab": "repetitive oscillation without development.",
        "aaba": "brief spike that fails to sustain interest.",
        "edcb": "front-loaded novelty trailing off — the 'important part is over' signal.",
        "dcba": "steady decline from high to low — reader attention fading.",
        "edca": "sharp drop from peak — premature resolution.",
    }
    return interpretations.get(motif, "a pacing pattern to avoid for this genre.")
