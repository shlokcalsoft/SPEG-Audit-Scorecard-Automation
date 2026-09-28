import json

import pytest

from src.assessment_engine import (
    Assessment,
    assess,
    enforce_rules,
    no_evidence_assessment,
    parse_assessment,
    rule_based_assessment,
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


def test_assess_reads_extractor_items_and_carries_quotes_forward(tmp_path, monkeypatch):
    import src.assessment_engine as engine

    registry_path = tmp_path / "registry.json"
    evidence_path = tmp_path / "evidence.json"
    output_path = tmp_path / "assessments.json"
    registry_path.write_text(json.dumps({"checkpoints": [{
        "checkpoint_id": "C001", "phase": "Initiation", "activity": "SOW",
        "checkpoint_text": "Approved SOW is available",
    }]}), encoding="utf-8")
    record = evidence_record("AMBIGUOUS")
    record["items"] = record.pop("evidence")
    evidence_path.write_text(json.dumps({"evidence": [record]}), encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_ONLY", "true")
    monkeypatch.setattr(engine, "request_json", lambda *args, **kwargs: json.dumps({
        "response": "Partial", "observation": "Evidence discusses the SOW but does not confirm approval.",
        "confidence": 0.8, "needs_review": True, "review_reason": "Approval is unclear.",
        "evidence_status": "AMBIGUOUS", "evidence": [],
    }))

    document = assess(registry_path, evidence_path, output_path)
    result = document["assessments"][0]

    assert result["response"] == "Partial"
    assert result["evidence_status"] == "AMBIGUOUS"
    assert result["evidence"][0]["unit_id"] == "WIN0001"
    assert result["needs_review"] is True


def test_rule_fallback_suggests_no_from_clear_found_contradiction():
    record = evidence_record("FOUND")
    record["items"] = record.pop("evidence")
    record["items"][0]["stance"] = "contradicts"

    result = rule_based_assessment(record, "Model unavailable")

    assert result.response == "No"
    assert result.needs_review is True
    assert result.evidence[0].stance == "contradicts"


def test_rule_fallback_uses_partial_for_ambiguous_evidence():
    record = evidence_record("AMBIGUOUS")
    record["items"] = record.pop("evidence")
    record["items"][0]["stance"] = "clarifies"

    result = rule_based_assessment(record)

    assert result.response == "Partial"
    assert result.needs_review is True