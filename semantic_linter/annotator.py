"""
Pacing Annotator — Overlay pacing lint results on manuscript PDFs.

Takes a PacingReport and the source PDF, then adds visual annotations:
- Highlight bars in the margin showing novelty level (color-coded SAX)
- Comment boxes at flagged locations with issue + suggestion
- A summary dashboard on an appended page

This is an OPTIONAL post-linting step in the pipeline.

Usage:
    from codexes.modules.rkhs.pacing_linter import PacingLinter
    from codexes.modules.rkhs.pacing_annotator import PacingAnnotator

    linter = PacingLinter()
    report = linter.lint_text(text, profile="risk_atlas")
    annotator = PacingAnnotator()
    annotator.annotate_pdf("input.pdf", report, "output_annotated.pdf")
"""

import fitz  # PyMuPDF
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .linter import PacingReport, PacingFlag, Severity, ScalarDynamics


# ---------------------------------------------------------------------------
# Color scheme — SAX level to color
# ---------------------------------------------------------------------------

SAX_COLORS = {
    'a': (0.2, 0.4, 0.8),     # blue — low novelty
    'b': (0.3, 0.7, 0.5),     # teal — below average
    'c': (0.5, 0.8, 0.3),     # green — average
    'd': (0.9, 0.7, 0.1),     # amber — above average
    'e': (0.9, 0.2, 0.2),     # red — high novelty
}

SEVERITY_COLORS = {
    Severity.ERROR:   (0.9, 0.1, 0.1),    # red
    Severity.WARNING: (0.9, 0.6, 0.0),    # orange
    Severity.INFO:    (0.2, 0.5, 0.8),    # blue
}

SEVERITY_LABELS = {
    Severity.ERROR:   "ERROR",
    Severity.WARNING: "WARN",
    Severity.INFO:    "INFO",
}


# ---------------------------------------------------------------------------
# Annotator
# ---------------------------------------------------------------------------

class PacingAnnotator:
    """Annotates a PDF with pacing lint results."""

    def __init__(self, margin_width: float = 12.0, comment_width: float = 180.0):
        """
        Args:
            margin_width: Width of the SAX novelty bar in the margin (points)
            comment_width: Width of comment annotation boxes (points)
        """
        self.margin_width = margin_width
        self.comment_width = comment_width

    def annotate_pdf(self, input_pdf: str, report: PacingReport,
                     output_pdf: str, add_dashboard: bool = True) -> Path:
        """
        Annotate a PDF with pacing lint results.

        Args:
            input_pdf: Path to source manuscript PDF
            report: PacingReport from PacingLinter
            output_pdf: Path for annotated output PDF
            add_dashboard: Whether to append a summary dashboard page

        Returns:
            Path to the annotated PDF
        """
        doc = fitz.open(input_pdf)
        n_pages = len(doc)

        if n_pages == 0:
            raise ValueError("PDF has no pages")

        # --- 1. SAX novelty margin bars ---
        self._add_novelty_bars(doc, report)

        # --- 2. Flag annotations ---
        self._add_flag_annotations(doc, report)

        # --- 3. Dashboard page ---
        if add_dashboard:
            self._add_dashboard_page(doc, report)

        # Save
        output_path = Path(output_pdf)
        doc.save(str(output_path), garbage=4, deflate=True)
        doc.close()
        return output_path

    def _add_novelty_bars(self, doc: fitz.Document, report: PacingReport):
        """Add colored margin bars showing novelty level per page region."""
        n_pages = len(doc)
        sax_str = report.sax_string
        if not sax_str:
            return

        # Map SAX segments to pages proportionally
        segments_per_page = max(1, len(sax_str) / n_pages)

        for page_num in range(n_pages):
            page = doc[page_num]
            rect = page.rect
            bar_x = rect.width - self.margin_width - 3  # right margin

            # Which SAX segments cover this page?
            seg_start = int(page_num * segments_per_page)
            seg_end = min(int((page_num + 1) * segments_per_page), len(sax_str))

            if seg_start >= len(sax_str):
                continue

            # Draw one bar per SAX segment that maps to this page
            n_segs_on_page = max(1, seg_end - seg_start)
            bar_height = (rect.height - 72) / n_segs_on_page  # 36pt margins

            for i, seg_idx in enumerate(range(seg_start, seg_end)):
                if seg_idx >= len(sax_str):
                    break
                letter = sax_str[seg_idx]
                color = SAX_COLORS.get(letter, (0.5, 0.5, 0.5))

                y_top = 36 + i * bar_height
                y_bot = y_top + bar_height - 1

                bar_rect = fitz.Rect(bar_x, y_top, bar_x + self.margin_width, y_bot)
                shape = page.new_shape()
                shape.draw_rect(bar_rect)
                shape.finish(color=color, fill=color, fill_opacity=0.6)
                shape.commit()

                # Label the SAX letter
                label_point = fitz.Point(bar_x + 2, y_top + bar_height / 2 + 3)
                page.insert_text(
                    label_point, letter.upper(),
                    fontsize=7, fontname="helv",
                    color=(1, 1, 1),
                )

    def _add_flag_annotations(self, doc: fitz.Document, report: PacingReport):
        """Add comment annotations for each pacing flag."""
        n_pages = len(doc)
        if not report.flags:
            return

        # Group flags by approximate page location
        # "Whole text" flags go on page 1
        # "Chapter N" flags go proportionally through the document
        # "Paragraphs X-Y" flags map by paragraph position

        n_paras = len(report.novelty_curve) + 1 if len(report.novelty_curve) > 0 else 1

        for flag in report.flags:
            page_num = self._flag_to_page(flag, n_pages, n_paras)
            if page_num >= n_pages:
                page_num = n_pages - 1

            page = doc[page_num]
            rect = page.rect

            sev_color = SEVERITY_COLORS[flag.severity]
            sev_label = SEVERITY_LABELS[flag.severity]

            # Create a text annotation (sticky note style)
            comment_text = (
                f"[{sev_label}] {flag.issue}\n\n"
                f"Suggestion: {flag.suggestion}"
            )

            # Position comments in the left margin, staggered to avoid overlap
            y_pos = 72 + (hash(flag.issue) % 600)
            annot_rect = fitz.Rect(5, y_pos, 30, y_pos + 25)

            annot = page.add_text_annot(
                annot_rect.tl,
                comment_text,
                icon="Comment",
            )
            annot.set_colors(stroke=sev_color)
            annot.update()

    def _add_dashboard_page(self, doc: fitz.Document, report: PacingReport):
        """Append a summary dashboard page."""
        # Create a new page (letter size)
        page = doc.new_page(width=612, height=792)

        y = 50  # cursor

        # Title
        page.insert_text(fitz.Point(50, y), "PACING LINT REPORT",
                        fontsize=18, fontname="helv", color=(0.1, 0.1, 0.1))
        y += 25
        page.insert_text(fitz.Point(50, y), f"Profile: {report.profile_name}",
                        fontsize=11, fontname="helv", color=(0.3, 0.3, 0.3))
        y += 30

        # Scalars summary
        s = report.scalars
        page.insert_text(fitz.Point(50, y), "Scalar Dynamics",
                        fontsize=13, fontname="helv", color=(0.1, 0.1, 0.1))
        y += 18
        scalars_text = (
            f"Mean novelty: {s.mean_novelty:.4f}    "
            f"Std: {s.std_novelty:.4f}    "
            f"Speed: {s.speed:.4f}    "
            f"Circuitousness: {s.circuitousness:.1f}"
        )
        page.insert_text(fitz.Point(50, y), scalars_text,
                        fontsize=9, fontname="helv", color=(0.2, 0.2, 0.2))
        y += 25

        # SAX string
        page.insert_text(fitz.Point(50, y), "SAX String",
                        fontsize=13, fontname="helv", color=(0.1, 0.1, 0.1))
        y += 18

        # Draw SAX string as colored blocks
        sax = report.sax_string
        block_w = min(30, (512) / max(len(sax), 1))
        for i, letter in enumerate(sax):
            color = SAX_COLORS.get(letter, (0.5, 0.5, 0.5))
            bx = 50 + i * block_w
            block_rect = fitz.Rect(bx, y, bx + block_w - 1, y + 20)
            shape = page.new_shape()
            shape.draw_rect(block_rect)
            shape.finish(color=color, fill=color, fill_opacity=0.7)
            shape.commit()
            page.insert_text(fitz.Point(bx + block_w / 2 - 3, y + 14),
                           letter.upper(), fontsize=8, fontname="helv",
                           color=(1, 1, 1))
        y += 35

        # SAX legend
        legend_x = 50
        for letter, label in [('a', 'Low'), ('b', 'Below avg'), ('c', 'Average'),
                               ('d', 'Above avg'), ('e', 'High')]:
            color = SAX_COLORS[letter]
            lr = fitz.Rect(legend_x, y, legend_x + 12, y + 12)
            shape = page.new_shape()
            shape.draw_rect(lr)
            shape.finish(fill=color, fill_opacity=0.7)
            shape.commit()
            page.insert_text(fitz.Point(legend_x + 15, y + 10),
                           f"{letter.upper()}: {label}", fontsize=7,
                           fontname="helv", color=(0.3, 0.3, 0.3))
            legend_x += 80
        y += 25

        # Novelty curve visualization
        curve = report.novelty_curve
        if len(curve) > 2:
            page.insert_text(fitz.Point(50, y), "Novelty Curve",
                            fontsize=13, fontname="helv", color=(0.1, 0.1, 0.1))
            y += 15

            chart_x, chart_y = 50, y
            chart_w, chart_h = 512, 120

            # Draw axes
            shape = page.new_shape()
            shape.draw_line(fitz.Point(chart_x, chart_y + chart_h),
                          fitz.Point(chart_x + chart_w, chart_y + chart_h))
            shape.draw_line(fitz.Point(chart_x, chart_y),
                          fitz.Point(chart_x, chart_y + chart_h))
            shape.finish(color=(0.7, 0.7, 0.7), width=0.5)
            shape.commit()

            # Normalize curve to chart height
            c_min, c_max = np.min(curve), np.max(curve)
            c_range = c_max - c_min if c_max > c_min else 1.0

            # Draw curve as connected line segments
            shape = page.new_shape()
            points = []
            for i, val in enumerate(curve):
                px = chart_x + (i / (len(curve) - 1)) * chart_w
                py = chart_y + chart_h - ((val - c_min) / c_range) * (chart_h - 10)
                points.append(fitz.Point(px, py))

            if len(points) > 1:
                shape.draw_line(points[0], points[1])
                for i in range(2, len(points)):
                    shape.draw_line(points[i - 1], points[i])
                shape.finish(color=(0.2, 0.4, 0.8), width=1.0)
                shape.commit()

            # Y-axis labels
            page.insert_text(fitz.Point(chart_x - 2, chart_y + 8),
                           f"{c_max:.3f}", fontsize=6, fontname="helv",
                           color=(0.5, 0.5, 0.5))
            page.insert_text(fitz.Point(chart_x - 2, chart_y + chart_h - 2),
                           f"{c_min:.3f}", fontsize=6, fontname="helv",
                           color=(0.5, 0.5, 0.5))

            y += chart_h + 20

        # Flags summary
        page.insert_text(fitz.Point(50, y), "Pacing Flags",
                        fontsize=13, fontname="helv", color=(0.1, 0.1, 0.1))
        y += 18

        n_err = sum(1 for f in report.flags if f.severity == Severity.ERROR)
        n_warn = sum(1 for f in report.flags if f.severity == Severity.WARNING)
        n_info = sum(1 for f in report.flags if f.severity == Severity.INFO)
        grade = "PASS" if n_err == 0 and n_warn == 0 else ("WARN" if n_err == 0 else "FAIL")

        grade_color = (0.1, 0.7, 0.2) if grade == "PASS" else (
            (0.9, 0.6, 0.0) if grade == "WARN" else (0.9, 0.1, 0.1))

        page.insert_text(fitz.Point(50, y),
                        f"Grade: {grade}    Errors: {n_err}    "
                        f"Warnings: {n_warn}    Info: {n_info}",
                        fontsize=11, fontname="helv", color=grade_color)
        y += 20

        # List each flag
        for flag in report.flags:
            if y > 740:
                # New page if we run out of room
                page = doc.new_page(width=612, height=792)
                y = 50
                page.insert_text(fitz.Point(50, y), "Pacing Flags (continued)",
                                fontsize=13, fontname="helv", color=(0.1, 0.1, 0.1))
                y += 20

            sev_color = SEVERITY_COLORS[flag.severity]
            sev_label = SEVERITY_LABELS[flag.severity]

            # Severity indicator dot
            shape = page.new_shape()
            shape.draw_circle(fitz.Point(55, y - 3), 3)
            shape.finish(fill=sev_color, fill_opacity=0.8)
            shape.commit()

            page.insert_text(fitz.Point(63, y),
                           f"[{sev_label}] {flag.location}: {flag.issue[:75]}",
                           fontsize=7.5, fontname="helv", color=(0.2, 0.2, 0.2))
            y += 11
            page.insert_text(fitz.Point(63, y),
                           f"  -> {flag.suggestion[:80]}",
                           fontsize=7, fontname="helv", color=(0.4, 0.4, 0.4))
            y += 14

        # Chapter SAX strings
        if report.chapter_sax:
            y += 10
            if y > 700:
                page = doc.new_page(width=612, height=792)
                y = 50
            page.insert_text(fitz.Point(50, y), "Per-Chapter SAX",
                            fontsize=13, fontname="helv", color=(0.1, 0.1, 0.1))
            y += 18
            for i, ch_sax in enumerate(report.chapter_sax):
                if y > 760:
                    break
                page.insert_text(fitz.Point(50, y),
                               f"Ch {i+1}: {ch_sax}",
                               fontsize=8, fontname="helv", color=(0.3, 0.3, 0.3))
                y += 12

    @staticmethod
    def _flag_to_page(flag: PacingFlag, n_pages: int, n_paras: int) -> int:
        """Map a flag's location to an approximate page number."""
        loc = flag.location.lower()

        if "whole text" in loc:
            return 0

        # "Chapter N" → proportional
        import re
        ch_match = re.search(r'chapter\s+(\d+)', loc)
        if ch_match:
            ch_num = int(ch_match.group(1))
            # Rough: assume chapters are evenly distributed
            return min(int(ch_num / max(ch_num + 2, 1) * n_pages), n_pages - 1)

        # "Paragraphs X-Y" → proportional
        para_match = re.search(r'paragraphs?\s+(\d+)', loc)
        if para_match:
            para_num = int(para_match.group(1))
            return min(int(para_num / n_paras * n_pages), n_pages - 1)

        # "SAX segments" → proportional
        seg_match = re.search(r'segments?\s+(\d+)', loc)
        if seg_match:
            seg_num = int(seg_match.group(1))
            return min(int(seg_num / 16 * n_pages), n_pages - 1)

        return 0


def annotate_from_report(input_pdf: str, report: PacingReport,
                         output_pdf: Optional[str] = None) -> Path:
    """
    Convenience function: annotate a PDF with a pacing report.

    If output_pdf is None, appends '_pacing_annotated' to the input filename.
    """
    if output_pdf is None:
        p = Path(input_pdf)
        output_pdf = str(p.parent / f"{p.stem}_pacing_annotated{p.suffix}")

    annotator = PacingAnnotator()
    return annotator.annotate_pdf(input_pdf, report, output_pdf)
