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
from .provider_router import request_fallback, request_with_fallback


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
MAX_OUTPUT_TOKENS = int(os.getenv("ASSESSMENT_MAX_OUTPUT_TOKENS", "450"))
SCHEMA_VERSION = 1

EvidenceStatus = Literal["FOUND", "NO_EVIDENCE", "AMBIGUOUS", "CONFLICTING"]
ResponseValue = Literal["Yes", "Partial", "In Progress", "No", "NA", "Not Clarified"]
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
The response must be exactly one of: Yes, Partial, In Progress, No, NA, Not Clarified.
NO_EVIDENCE must always have response null or Not Clarified, needs_review true, and confidence below 0.6.
AMBIGUOUS and CONFLICTING evidence must require review.
Use Yes only when the evidence clearly establishes the checkpoint.
Use Partial when evidence establishes only part of the checkpoint.
Use No only when evidence clearly contradicts the checkpoint; absence of evidence is never No.
Use NA only when the checkpoint is explicitly not applicable in the supplied evidence.

Return JSON only:
{"response":"Yes|Partial|In Progress|No|NA|Not Clarified|null","observation":"...","confidence":0.0,"needs_review":true,"review_reason":"... or null"}"""


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
    evidence = evidence_record.get("evidence") or evidence_record.get("items", [])
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
        evidence_status=(
            evidence_record.get("evidence_status", "NO_EVIDENCE")
            if evidence_record.get("extraction_error")
            else "NO_EVIDENCE"
        ),
        evidence=[],
    )


def rule_based_assessment(evidence_record: dict[str, Any], reason: str | None = None) -> Assessment:
    """Create a conservative, review-required draft from validated evidence stances."""
    items = evidence_items(evidence_record)
    stances = {item.get("stance") for item in items}
    status = evidence_record.get("evidence_status", "AMBIGUOUS")
    if not items or status == "NO_EVIDENCE":
        return no_evidence_assessment(evidence_record)
    if status == "FOUND" and stances == {"supports"}:
        response: ResponseValue = "Yes"
    elif status == "FOUND" and stances == {"contradicts"}:
        response = "No"
    else:
        response = "Partial"
    detail = reason or "OpenRouter assessment was unavailable; response was drafted from validated evidence stances."
    return Assessment(
        response=response,
        observation=evidence_record.get("observation") or "Validated transcript evidence requires reviewer confirmation.",
        confidence=min(float(evidence_record.get("confidence", 0.0)), 0.59),
        needs_review=True,
        review_reason=detail,
        evidence_status=status,
        evidence=[AssessmentEvidence(**item) for item in items],
    )


def evidence_items(evidence_record: dict[str, Any]) -> list[dict[str, Any]]:
    """Read validated quote items from either pipeline artifact contract."""
    return evidence_record.get("evidence") or evidence_record.get("items", [])


def request_json(client: Any, model: str, messages: list[dict[str, str]]) -> str:
    """Request JSON, retrying without Groq JSON mode for incompatible models."""
    try:
        content, _provider = request_with_fallback(client, model, messages, MAX_OUTPUT_TOKENS)
        return content
    except Exception as error:
        message = str(error).lower()
        if "json_validate_failed" not in message and "response_format" not in message:
            raise
    response = client.chat.completions.create(
        model=model, temperature=0, max_tokens=MAX_OUTPUT_TOKENS, messages=messages,
    )
    content = response.choices[0].message.content
    if not content:
        return request_fallback(messages, MAX_OUTPUT_TOKENS)[0]
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
    items = evidence_items(evidence_record)
    if evidence_status == "NO_EVIDENCE" or not items:
        return no_evidence_assessment(evidence_record)

    assessment.evidence_status = evidence_status
    assessment.evidence = [
        AssessmentEvidence(**item) for item in items
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
    start: int = 1,
    end: int | None = None,
    local_only: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    """Assess every checkpoint from evidence results and write assessments JSON."""
    registry = load_json(registry_path)
    evidence_document = load_json(evidence_path)
    checkpoints = registry.get("checkpoints", registry)
    evidence_by_id = {
        item["checkpoint_id"]: item for item in evidence_document.get("evidence", [])
    }
    if limit is not None:
        selected = checkpoints[:limit]
    else:
        selected = checkpoints[start - 1:end]
    injected_client = client is not None
    needs_client = any(
        evidence_items(evidence_by_id.get(checkpoint["checkpoint_id"], {}))
        and evidence_by_id.get(checkpoint["checkpoint_id"], {}).get("evidence_status") != "NO_EVIDENCE"
        for checkpoint in selected
    )
    if client is None and needs_client and not local_only:
        openrouter_only = os.getenv("OPENROUTER_ONLY", "false").lower() == "true"
        if not openrouter_only and not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is required to assess evidence-bearing checkpoints")
        if openrouter_only:
            client = object()
            model = model or "openrouter"
        else:
            try:
                from groq import Groq
            except ImportError as exc:
                raise RuntimeError("Install Groq with: py -m pip install groq") from exc
            client = Groq(api_key=os.environ["GROQ_API_KEY"])
            model = resolve_model(client, model)
    elif injected_client:
        model = model or "injected-test-model"
    elif local_only:
        model = "local-rules"

    existing_records = load_json(output_path).get("assessments", []) if output_path.exists() else []
    existing_by_id = {
        record["checkpoint_id"]: record
        for record in existing_records
        if record.get("checkpoint_id")
    }
    for checkpoint in selected:
        evidence_record = evidence_by_id.get(
            checkpoint["checkpoint_id"],
            {"evidence_status": "NO_EVIDENCE", "evidence": [], "review_reason": "No evidence record exists."},
        )
        if not evidence_items(evidence_record) or evidence_record.get("evidence_status") == "NO_EVIDENCE":
            assessment = no_evidence_assessment(evidence_record)
            assessment_method = "no-evidence-rule"
        elif local_only:
            assessment = rule_based_assessment(evidence_record)
            assessment_method = "local-evidence-rule"
        else:
            try:
                if client is None:
                    raise RuntimeError("No assessment client is configured")
                assessment = enforce_rules(
                    assess_response(client, checkpoint, evidence_record, model), evidence_record
                )
                assessment_method = "model"
            except Exception as error:
                assessment = rule_based_assessment(evidence_record, f"Model assessment failed: {error}")
                assessment_method = "local-evidence-rule-fallback"
        existing_by_id[checkpoint["checkpoint_id"]] = {
            "checkpoint_id": checkpoint["checkpoint_id"],
            **_dump(assessment),
            "assessment_method": assessment_method,
        }
        partial_records = [
            existing_by_id[item["checkpoint_id"]]
            for item in checkpoints
            if item["checkpoint_id"] in existing_by_id
        ]
        partial_payload = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "prompt_hash": prompt_hash(),
            "checkpoint_count": len(partial_records),
            "assessments": partial_records,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(partial_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    merged = existing_by_id
    records = [
        merged[checkpoint["checkpoint_id"]]
        for checkpoint in checkpoints
        if checkpoint["checkpoint_id"] in merged
    ]
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
    parser.add_argument("--start", type=int, default=1, help="First checkpoint number, inclusive")
    parser.add_argument("--end", type=int, help="Last checkpoint number, exclusive")
    parser.add_argument("--local-only", action="store_true", help="Use deterministic evidence rules without calling a model")
    args = parser.parse_args()
    document = assess(model=args.model, limit=args.limit, start=args.start, end=args.end, local_only=args.local_only)
    print(f"Wrote {document['checkpoint_count']} assessments to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()