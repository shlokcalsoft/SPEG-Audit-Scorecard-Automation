import json

import pytest

from src.assessment_engine import (
    Assessment,
    assess,
    enforce_rules,
    no_evidence_assessment,
    parse_assessment,
)


def evidence_record(status="FOUND"):
    return {
        "checkpoint_id": "C001",
        "evidence_status": status,
        "evidence": [{
            "unit_id": "WIN0001",
            "segment_ids": ["SEG1"],
            "timestamp": "0:01",
            "speaker": "Owner",
            "verbatim_quote": "The approved SOW is stored in SharePoint.",
            "stance": "supports",
            "reason": "The owner states the artifact location.",
        }],
        "observation": "The approved SOW was discussed.",
        "confidence": 0.9,
        "needs_review": False,
        "review_reason": None,
    }


def model_assessment(response="Yes", confidence=0.9):
    return Assessment(
        response=response,
        observation="The evidence supports availability.",
        confidence=confidence,
        needs_review=False,
        review_reason=None,
        evidence_status="FOUND",
        evidence=[],
    )


def test_no_evidence_never_becomes_no():
    result = no_evidence_assessment({
        "evidence_status": "NO_EVIDENCE",
        "evidence": [],
        "observation": "Nothing was discussed.",
    })

    assert result.response == "Not Clarified"
    assert result.needs_review is True
    assert result.confidence == 0.0


def test_conflicting_evidence_requires_review():
    record = evidence_record("CONFLICTING")
    result = enforce_rules(model_assessment(), record)

    assert result.evidence_status == "CONFLICTING"
    assert result.needs_review is True


def test_low_confidence_requires_review():
    result = enforce_rules(model_assessment(confidence=0.4), evidence_record())

    assert result.needs_review is True
    assert result.review_reason


def test_ambiguous_no_response_is_preserved_for_review():
    result = enforce_rules(model_assessment(response="No"), evidence_record("AMBIGUOUS"))

    assert result.response == "Not Clarified"
    assert result.needs_review is True


def test_assess_writes_records_and_skips_api_for_no_evidence(tmp_path):
    registry_path = tmp_path / "registry.json"
    evidence_path = tmp_path / "evidence.json"
    output_path = tmp_path / "assessments.json"
    registry_path.write_text(json.dumps({"checkpoints": [{"checkpoint_id": "C001"}]}), encoding="utf-8")
    evidence_path.write_text(json.dumps({"evidence": [{
        "checkpoint_id": "C001", "evidence_status": "NO_EVIDENCE", "evidence": [],
        "observation": "No evidence.", "confidence": 0.0, "needs_review": True,
    }]}), encoding="utf-8")

    document = assess(registry_path, evidence_path, output_path)

    assert document["assessments"][0]["response"] == "Not Clarified"
    assert output_path.exists()