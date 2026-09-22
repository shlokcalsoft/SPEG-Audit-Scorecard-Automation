"""Convert validated transcript evidence into auditable checkpoint assessments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / "data" / "checkpoint_registry.json"
EVIDENCE_PATH = PROJECT_ROOT / "data" / "evidence_results.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "assessments.json"
load_dotenv(PROJECT_ROOT / ".env")
MODEL = os.getenv("GROQ_MODEL")
MODEL_PREFERENCES = (
    "openai/gpt-oss-20b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2-instruct",
    "llama-3.1-8b-instant",
)
MIN_CONFIDENCE = 0.6
SCHEMA_VERSION = 1

EvidenceStatus = Literal["FOUND", "NO_EVIDENCE", "AMBIGUOUS", "CONFLICTING"]
ResponseValue = Literal["Yes", "Partial", "No", "NA", "Not Clarified"]
Stance = Literal["supports", "contradicts", "clarifies", "supersedes"]


class AssessmentEvidence(BaseModel):
    """Evidence carried forward into the assessment artifact."""

    unit_id: str
    segment_ids: list[str]
    timestamp: str | None = None
    speaker: str | None = None
    verbatim_quote: str
    stance: Stance


class Assessment(BaseModel):
    """One checkpoint assessment with explicit review state."""

    response: ResponseValue | None = None
    observation: str
    confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool
    review_reason: str | None = None
    evidence_status: EvidenceStatus
    evidence: list[AssessmentEvidence] = Field(default_factory=list)


SYSTEM_PROMPT = """You are a conservative SEPG audit assessment engine.
Assess only the supplied checkpoint and already-validated evidence.
Do not add, rewrite, or invent evidence. Do not use outside knowledge.
The response must be exactly one of: Yes, Partial, No, NA, Not Clarified.
NO_EVIDENCE must always have response null or Not Clarified, needs_review true, and confidence below 0.6.
AMBIGUOUS and CONFLICTING evidence must require review.
Use Yes only when the evidence clearly establishes the checkpoint.
Use Partial when evidence establishes only part of the checkpoint.
Use No only when evidence clearly contradicts the checkpoint; absence of evidence is never No.
Use NA only when the checkpoint is explicitly not applicable in the supplied evidence.

Return JSON only:
{"response":"Yes|Partial|No|NA|Not Clarified|null","observation":"...","confidence":0.0,"needs_review":true,"review_reason":"... or null"}"""


def load_json(path: Path) -> Any:
    """Load a JSON artifact."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_model(client: Any, requested_model: str | None) -> str:
    """Use the requested model or select an available Groq text model."""
    available = {item.id for item in client.models.list().data}
    if requested_model and requested_model in available:
        return requested_model
    for preferred in MODEL_PREFERENCES:
        if preferred in available:
            return preferred
    candidates = sorted(
        model_id for model_id in available
        if "whisper" not in model_id.lower() and "guard" not in model_id.lower()
    )
    if not candidates:
        raise RuntimeError("Groq returned no usable text models for this account")
    return candidates[0]


def _dump(model: BaseModel) -> dict[str, Any]:
    """Serialize with either Pydantic v1 or v2."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _validate_json(model_type: type[BaseModel], content: str) -> BaseModel:
    """Validate JSON with either Pydantic v1 or v2."""
    try:
        if hasattr(model_type, "model_validate_json"):
            return model_type.model_validate_json(content)
        return model_type.parse_raw(content)
    except (ValidationError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid assessment response: {exc}") from exc


def parse_assessment(response_text: str) -> Assessment:
    """Parse one model assessment response into the public schema."""
    response_text = response_text.strip()
    if response_text.startswith("```"):
        import re
        response_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response_text, flags=re.IGNORECASE).strip()
    return _validate_json(Assessment, response_text)


def prompt_hash() -> str:
    """Return a stable hash of the assessment instructions."""
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def build_prompt(checkpoint: dict[str, Any], evidence_record: dict[str, Any]) -> str:
    """Build an assessment prompt from validated evidence only."""
    evidence = evidence_record.get("evidence", [])
    evidence_text = "\n\n".join(
        f"UNIT: {item['unit_id']} | SEGMENTS: {', '.join(item.get('segment_ids', []))}\n"
        f"STANCE: {item['stance']}\nQUOTE: {item['verbatim_quote']}\n"
        f"EVIDENCE REASON: {item.get('reason', '')}"
        for item in evidence
    ) or "No validated evidence items."
    return (
        f"CHECKPOINT ID: {checkpoint['checkpoint_id']}\n"
        f"PHASE: {checkpoint.get('phase', '')}\n"
        f"ACTIVITY: {checkpoint.get('activity', '')}\n"
        f"REQUIREMENT: {checkpoint.get('checkpoint_text', '')}\n"
        f"EVIDENCE STATUS: {evidence_record.get('evidence_status', 'NO_EVIDENCE')}\n\n"
        f"VALIDATED EVIDENCE:\n{evidence_text}"
    )


def no_evidence_assessment(evidence_record: dict[str, Any]) -> Assessment:
    """Create the safe assessment without spending an LLM call."""
    reason = evidence_record.get("review_reason") or "No validated evidence was found."
    return Assessment(
        response="Not Clarified",
        observation=evidence_record.get("observation") or "The checkpoint was not clarified in the transcript.",
        confidence=0.0,
        needs_review=True,
        review_reason=reason,
        evidence_status="NO_EVIDENCE",
        evidence=[],
    )


def request_json(client: Any, model: str, messages: list[dict[str, str]]) -> str:
    """Request JSON, retrying without Groq JSON mode for incompatible models."""
    try:
        response = client.chat.completions.create(
            model=model, temperature=0, max_tokens=700,
            response_format={"type": "json_object"}, messages=messages,
        )
        content = response.choices[0].message.content
        if content:
            return content
    except Exception as error:
        message = str(error).lower()
        if "json_validate_failed" not in message and "response_format" not in message:
            raise
    response = client.chat.completions.create(
        model=model, temperature=0, max_tokens=700, messages=messages,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Groq returned an empty assessment response")
    return content


def assess_response(client: Any, checkpoint: dict[str, Any], evidence_record: dict[str, Any], model: str) -> Assessment:
    """Ask Groq to assess one non-empty evidence record."""
    content = request_json(
        client, model, [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(checkpoint, evidence_record)},
        ],
    )
    parsed = _validate_json(Assessment, content)
    return parsed


def enforce_rules(assessment: Assessment, evidence_record: dict[str, Any]) -> Assessment:
    """Apply non-negotiable assessment rules after model output."""
    evidence_status = evidence_record.get("evidence_status", "NO_EVIDENCE")
    if evidence_status == "NO_EVIDENCE" or not evidence_record.get("evidence"):
        return no_evidence_assessment(evidence_record)

    assessment.evidence_status = evidence_status
    assessment.evidence = [
        AssessmentEvidence(**item) for item in evidence_record.get("evidence", [])
    ]
    if evidence_status in {"AMBIGUOUS", "CONFLICTING"}:
        assessment.needs_review = True
        assessment.review_reason = assessment.review_reason or (
            f"Evidence status is {evidence_status.lower()}."
        )
    if assessment.confidence < MIN_CONFIDENCE:
        assessment.needs_review = True
        assessment.review_reason = assessment.review_reason or "Confidence is below the review threshold."
    if assessment.response == "No" and evidence_status != "FOUND":
        assessment.response = "Not Clarified"
        assessment.needs_review = True
        assessment.review_reason = assessment.review_reason or "A No response requires clear evidence."
    return assessment


def assess(
    registry_path: Path = REGISTRY_PATH,
    evidence_path: Path = EVIDENCE_PATH,
    output_path: Path = OUTPUT_PATH,
    model: str | None = MODEL,
    limit: int | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Assess every checkpoint from evidence results and write assessments JSON."""
    registry = load_json(registry_path)
    evidence_document = load_json(evidence_path)
    checkpoints = registry.get("checkpoints", registry)
    evidence_by_id = {
        item["checkpoint_id"]: item for item in evidence_document.get("evidence", [])
    }
    selected = checkpoints[:limit] if limit is not None else checkpoints
    injected_client = client is not None
    needs_client = any(
        evidence_by_id.get(checkpoint["checkpoint_id"], {}).get("evidence")
        and evidence_by_id.get(checkpoint["checkpoint_id"], {}).get("evidence_status") != "NO_EVIDENCE"
        for checkpoint in selected
    )
    if client is None and needs_client:
        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is required to assess evidence-bearing checkpoints")
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError("Install Groq with: py -m pip install groq") from exc
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        model = resolve_model(client, model)
    elif injected_client:
        model = model or "injected-test-model"

    records = []
    for checkpoint in selected:
        evidence_record = evidence_by_id.get(
            checkpoint["checkpoint_id"],
            {"evidence_status": "NO_EVIDENCE", "evidence": [], "review_reason": "No evidence record exists."},
        )
        if not evidence_record.get("evidence") or evidence_record.get("evidence_status") == "NO_EVIDENCE":
            assessment = no_evidence_assessment(evidence_record)
        else:
            if client is None:
                raise RuntimeError("An assessment client is required for evidence-bearing checkpoints")
            assessment = enforce_rules(
                assess_response(client, checkpoint, evidence_record, model), evidence_record
            )
        records.append({"checkpoint_id": checkpoint["checkpoint_id"], **_dump(assessment)})

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "prompt_hash": prompt_hash(),
        "checkpoint_count": len(records),
        "assessments": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    """Run the assessment engine from evidence JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    document = assess(model=args.model, limit=args.limit)
    print(f"Wrote {document['checkpoint_count']} assessments to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()