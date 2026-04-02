# semantic-linter

SAX-based narrative pacing quality control for longform text.

Computes inter-paragraph novelty curves from text embeddings, applies Symbolic Aggregate approXimation (SAX), and compares motif distributions against genre-specific target profiles. Produces actionable revision diagnostics.

## Install

```bash
pip install semantic-linter
```

For PDF annotation support:
```bash
pip install "semantic-linter[pdf]"
```

## Usage

### CLI

```bash
# Lint a text file
semantic-lint manuscript.md --profile risk_atlas

# Generate a punch list for LLM revision
semantic-lint chapter.md --profile middle_grade_fiction --punch-list revisions.json

# Annotate a PDF with pacing overlays
semantic-lint chapter.md --profile risk_atlas --annotate-pdf book.pdf --output-pdf annotated.pdf
```

### Python

```python
from semantic_linter import PacingLinter

linter = PacingLinter()
report = linter.lint_text(text, profile="risk_atlas")
print(report.summary())

# Generate revision punch list for LLM consumption
punch_list = report.punch_list()

# Annotate a PDF
from semantic_linter import annotate_from_report
annotate_from_report("input.pdf", report, "output_annotated.pdf")
```

## Profiles

Three built-in profiles:

| Profile | Audience | Key characteristics |
|---------|----------|-------------------|
| `middle_grade_fiction` | Ages 10-13 | Short wavelength, double-bump chapters, cliffhanger endings |
| `risk_atlas` | Policy analysts | Briefing structure, high inter-section novelty, circuitousness OK |
| `literary_fiction` | Adult readers | Longer wavelength, tolerance for slow sections |

Custom profiles:

```python
from semantic_linter import PacingProfile, PacingLinter

profile = PacingProfile(
    name="my_profile",
    description="Custom pacing target",
    mean_novelty_range=(0.28, 0.40),
    preferred_motifs=["bcdc", "cdcb"],
    avoided_motifs=["aaaa", "bbbb"],
)
linter = PacingLinter()
report = linter.lint_text(text, profile=profile)
```

## How it works

1. Split text into substantive paragraphs (filtering headings, numbers, short labels)
2. Embed paragraphs using Nomic Embed Text v1.5 (768 dimensions)
3. Compute inter-paragraph cosine distance (novelty curve)
4. Apply PAA + z-normalization + SAX discretization
5. Extract k-gram motifs from SAX string
6. Compare against profile: scalar ranges, motif distributions, chapter constraints
7. Generate flags with severity, location, and revision suggestions

## Citation

```bibtex
@article{zimmerman2026prescriptive,
  title={Prescriptive Pacing: SAX Motif Profiles as Generative Constraints
         for Audience-Optimized Narrative Dynamics},
  author={Zimmerman, Fred and Hilmar AI},
  year={2026}
}
```

## License

MIT
