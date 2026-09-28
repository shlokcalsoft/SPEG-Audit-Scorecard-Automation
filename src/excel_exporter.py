"""Write checkpoint assessments into a copy of the Excel scorecard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from openpyxl.styles import PatternFill


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = PROJECT_ROOT / "template" / "SEPG_Master_Audit_Template.xlsx"
REGISTRY_PATH = PROJECT_ROOT / "data" / "checkpoint_registry.json"
ASSESSMENTS_PATH = PROJECT_ROOT / "data" / "assessments.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "SEPG_Master_Audit_Completed.xlsx"
PROVIDER_FAILURE_MARKERS = (
    "provider request failed",
    "all configured openrouter models failed",
    "insufficient credits",
    "http 402",
    "http 429",
)

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def clean_review_note(note: str | None) -> str:
    """Keep operational provider diagnostics out of the scorecard cells."""
    if not note:
        return "Reviewer confirmation is required."
    if any(marker in note.lower() for marker in PROVIDER_FAILURE_MARKERS):
        return "AI service unavailable; review this checkpoint manually."
    return note[:500]


def export_assessments(
    template_path: Path = TEMPLATE_PATH,
    registry_path: Path = REGISTRY_PATH,
    assessments_path: Path = ASSESSMENTS_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """Fill mapped response, score, weighted-score, and observation cells."""
    registry_document = load_json(registry_path)
    assessment_document = load_json(assessments_path)
    checkpoints = registry_document.get("checkpoints", registry_document)
    assessments = {
        item["checkpoint_id"]: item
        for item in assessment_document.get("assessments", [])
    }

    workbook = load_workbook(template_path, data_only=False)
    written = 0
    observation_notes: dict[tuple[str, str], list[str]] = {}
    for checkpoint in checkpoints:
        assessment = assessments.get(checkpoint["checkpoint_id"])
        if assessment is None:
            continue
        worksheet = workbook[checkpoint["sheet"]]
        response = assessment.get("response") or "Not Clarified"
        worksheet[checkpoint["response_cell"]] = response
        observation = assessment.get("observation", "")
        if assessment.get("needs_review"):
            review_reason = clean_review_note(assessment.get("review_reason"))
            observation = f"{observation}\nREVIEW REQUIRED: {review_reason}"
            worksheet[checkpoint["response_cell"]].fill = PatternFill(
                fill_type="solid", fgColor="FFF2CC"
            )
        observation_cell = checkpoint["observation_cell"]
        for merged_range in worksheet.merged_cells.ranges:
            if observation_cell in merged_range:
                observation_cell = merged_range.start_cell.coordinate
                break
        observation_notes.setdefault((checkpoint["sheet"], observation_cell), []).append(
            f"[{checkpoint['checkpoint_id']}] {observation}"
        )
        written += 1

    for (sheet_name, observation_cell), notes in observation_notes.items():
        workbook[sheet_name][observation_cell] = "\n\n".join(notes)

    if hasattr(workbook, "calculation"):
        workbook.calculation.fullCalcOnLoad = True
        workbook.calculation.forceFullCalc = True
        workbook.calculation.calcMode = "auto"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)
    print(f"Wrote {written} assessments to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=TEMPLATE_PATH)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--assessments", type=Path, default=ASSESSMENTS_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    export_assessments(args.template, args.registry, args.assessments, args.output)


if __name__ == "__main__":
    main()