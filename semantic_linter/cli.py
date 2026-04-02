"""CLI entry point for semantic-linter."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="SAX-based narrative pacing linter for longform text"
    )
    parser.add_argument("input", help="Text file or directory to lint")
    parser.add_argument(
        "--profile", default="risk_atlas",
        choices=["middle_grade_fiction", "risk_atlas", "literary_fiction"],
        help="Pacing profile to lint against (default: risk_atlas)"
    )
    parser.add_argument(
        "--punch-list", metavar="PATH",
        help="Save punch list JSON to this path"
    )
    parser.add_argument(
        "--annotate-pdf", metavar="PDF",
        help="Annotate this PDF with lint results (requires pymupdf)"
    )
    parser.add_argument(
        "--output-pdf", metavar="PATH",
        help="Output path for annotated PDF (default: input_pacing_annotated.pdf)"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    from .linter import PacingLinter

    linter = PacingLinter()
    text = Path(args.input).read_text(encoding="utf-8", errors="replace")
    report = linter.lint_text(text, profile=args.profile)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())

    if args.punch_list:
        report.save_punch_list(Path(args.punch_list))
        print(f"\nPunch list saved to: {args.punch_list}", file=sys.stderr)

    if args.annotate_pdf:
        from .annotator import annotate_from_report
        out = args.output_pdf or None
        result = annotate_from_report(args.annotate_pdf, report, out)
        print(f"\nAnnotated PDF saved to: {result}", file=sys.stderr)


if __name__ == "__main__":
    main()
