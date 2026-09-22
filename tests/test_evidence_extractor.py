import json

import pytest

from src.evidence_extractor import (
    EvidenceResult,
    extract_evidence,
    get_checkpoint_context,
    index_units,
    parse_llm_response,
    validate_evidence_quotes,
)


def source_data():
    retrieval = {
        "results": [{
            "unit_id": "WIN0001", "unit_type": "window", "segment_ids": ["SEG1"],
            "unit_text": "Owner [0:01]: The approved SOW is stored in SharePoint.",
        }],
        "by_checkpoint": [{
            "checkpoint_id": "C001", "retrieval_status": "tagged",
            "tagged_units": [{"unit_id": "WIN0001"}], "fallback_units": [],
        }],
    }
    transcript = {"windows": [{
        "window_id": "WIN0001", "segment_ids": ["SEG1"],
        "text": "Owner [0:01]: The approved SOW is stored in SharePoint.",
    }]}
    return retrieval, transcript


def valid_result(quote="Owner [0:01]: The approved SOW is stored in SharePoint."):
    return parse_llm_response(json.dumps({
        "evidence_status": "FOUND", "items": [{
            "unit_id": "WIN0001", "segment_ids": ["SEG1"], "timestamp": "0:01",
            "speaker": "Owner", "verbatim_quote": quote, "stance": "supports", "reason": "The owner states it directly.",
        }], "observation": "The SOW is available.", "confidence": 0.9,
        "needs_review": False, "review_reason": None,
    }))


def test_indexes_retrieval_units_and_resolves_tagged_context():
    retrieval, transcript = source_data()
    units = index_units(retrieval, transcript)
    status, context = get_checkpoint_context("C001", retrieval, units)

    assert status == "tagged"
    assert context[0]["unit_id"] == "WIN0001"
    assert context[0]["segment_ids"] == ["SEG1"]


def test_fallback_units_are_used_when_tagged_is_empty():
    retrieval, transcript = source_data()
    retrieval["by_checkpoint"][0]["tagged_units"] = []
    retrieval["by_checkpoint"][0]["fallback_units"] = [{"unit_id": "WIN0001"}]
    status, context = get_checkpoint_context("C001", retrieval, index_units(retrieval, transcript))

    assert status == "fallback_only"
    assert len(context) == 1


def test_exact_and_whitespace_normalized_quotes_are_accepted():
    retrieval, transcript = source_data()
    units = index_units(retrieval, transcript).values()
    assert len(validate_evidence_quotes(valid_result(), list(units)).items) == 1
    result = valid_result("Owner [0:01]:   The approved SOW is stored in SharePoint.")
    assert len(validate_evidence_quotes(result, list(units)).items) == 1


def test_unmatched_quote_is_dropped_and_requires_review():
    retrieval, transcript = source_data()
    result = valid_result("Owner [0:01]: The database backup is complete.")
    validated = validate_evidence_quotes(result, list(index_units(retrieval, transcript).values()))

    assert validated.items == []
    assert validated.evidence_status == "NO_EVIDENCE"
    assert validated.needs_review is True


def test_mixed_stances_are_conflicting():
    retrieval, transcript = source_data()
    units = list(index_units(retrieval, transcript).values())
    result = valid_result()
    result.items.append(result.items[0].model_copy(update={"stance": "contradicts"}) if hasattr(result.items[0], "model_copy") else result.items[0].copy(update={"stance": "contradicts"}))
    validated = validate_evidence_quotes(result, units)

    assert validated.evidence_status == "CONFLICTING"
    assert validated.needs_review is True


def test_no_evidence_is_preserved_without_response_fields():
    result = EvidenceResult(evidence_status="NO_EVIDENCE", items=[], observation="Nothing found.", confidence=0.0, needs_review=True, review_reason="No evidence.")

    assert not hasattr(result, "response")
    assert result.items == []


def test_extract_evidence_writes_contract_with_mock_client(tmp_path):
    retrieval, transcript = source_data()
    registry = {"checkpoints": [{
        "checkpoint_id": "C001", "phase": "Initiation", "activity": "SOW",
        "checkpoint_text": "Approved SOW is available",
    }]}
    registry_path = tmp_path / "registry.json"
    retrieval_path = tmp_path / "retrieval.json"
    transcript_path = tmp_path / "transcript.json"
    output_path = tmp_path / "evidence.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    retrieval_path.write_text(json.dumps(retrieval), encoding="utf-8")
    transcript_path.write_text(json.dumps(transcript), encoding="utf-8")

    response_text = json.dumps({
        "evidence_status": "FOUND", "items": [{
            "unit_id": "WIN0001", "segment_ids": ["SEG1"], "timestamp": "0:01",
            "speaker": "Owner", "verbatim_quote": "Owner [0:01]: The approved SOW is stored in SharePoint.",
            "stance": "supports", "reason": "The source states the artifact is stored.",
        }], "observation": "The approved SOW is available.", "confidence": 0.9,
        "needs_review": False, "review_reason": None,
    })

    class Message:
        content = response_text

    class Choice:
        message = Message()

    class Completions:
        def create(self, **kwargs):
            return type("Response", (), {"choices": [Choice()]})()

    fake_client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    document = extract_evidence(registry_path, retrieval_path, transcript_path, output_path, client=fake_client)

    assert document["checkpoint_count"] == 1
    assert document["evidence"][0]["items"][0]["unit_id"] == "WIN0001"
    assert output_path.exists()
