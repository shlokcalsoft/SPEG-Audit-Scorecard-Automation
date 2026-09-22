"""Extract transcript-grounded evidence for each retrieved checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from rapidfuzz.fuzz import ratio
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / "data" / "checkpoint_registry.json"
RETRIEVAL_PATH = PROJECT_ROOT / "data" / "retrieval_results.json"
TRANSCRIPT_PATH = PROJECT_ROOT / "data" / "parsed_transcript.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "evidence_results.json"
load_dotenv(PROJECT_ROOT / ".env")
MODEL = os.getenv("GROQ_MODEL")
MODEL_PREFERENCES = (
    "openai/gpt-oss-20b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2-instruct",
    "llama-3.1-8b-instant",
)
MAX_UNITS = 8
MIN_CONFIDENCE = 0.6
SCHEMA_VERSION = 1


EvidenceStatus = Literal["FOUND", "NO_EVIDENCE", "AMBIGUOUS", "CONFLICTING"]
EvidenceStance = Literal["supports", "contradicts", "clarifies", "supersedes"]


class EvidenceItem(BaseModel):
    """One exact quote tied to one retrieved source unit."""

    unit_id: str
    segment_ids: list[str]
    timestamp: str | None = None
    speaker: str | None = None
    verbatim_quote: str
    stance: EvidenceStance
    reason: str


class EvidenceResult(BaseModel):
    """Validated evidence-only result for one checkpoint."""

    evidence_status: EvidenceStatus
    items: list[EvidenceItem] = Field(default_factory=list)
    observation: str
    confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool
    review_reason: str | None = None


SYSTEM_PROMPT = """You are a strict evidence extractor for a SEPG audit.
Use only the supplied checkpoint and source transcript units.
Do not use outside knowledge. Do not decide Yes, No, Partial, or NA.
Do not infer that a checkpoint is satisfied.
Every verbatim_quote must be copied exactly from one supplied source unit.
A question alone is not proof. Do not treat Yes, Yeah, Okay, Mhm, or similar acknowledgements
as evidence by themselves. If evidence is absent, return NO_EVIDENCE with an empty items array.

Return JSON with exactly these fields:
{
  "evidence_status": "FOUND|NO_EVIDENCE|AMBIGUOUS|CONFLICTING",
  "items": [{"unit_id":"...","segment_ids":["..."],"timestamp":"...","speaker":"...","verbatim_quote":"exact source text","stance":"supports|contradicts|clarifies|supersedes","reason":"..."}],
  "observation": "short transcript-grounded observation",
  "confidence": 0.0,
  "needs_review": true,
  "review_reason": "... or null"
}"""


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


def normalize_text(text: str) -> str:
    """Normalize whitespace and case for quote comparison only."""
    return " ".join((text or "").split()).lower()


def quote_matches(quote: str, source_text: str) -> bool:
    """Prefer exact substring matching and use fuzzy matching only as fallback."""
    normalized_quote = normalize_text(quote)
    normalized_source = normalize_text(source_text)
    if not normalized_quote or not normalized_source:
        return False
    if normalized_quote in normalized_source:
        return True
    return ratio(normalized_quote, normalized_source) >= 90


def timestamp_key(timestamp: str | None) -> tuple[int, str]:
    """Create a stable chronological sort key for M:SS or H:MM:SS."""
    if not timestamp:
        return (10**9, "")
    values = timestamp.split(":")
    if not all(value.isdigit() for value in values):
        return (10**9, timestamp)
    numbers = [int(value) for value in values]
    seconds = numbers[-1] + 60 * numbers[-2]
    if len(numbers) == 3:
        seconds += 3600 * numbers[-3]
    return (seconds, timestamp)


def index_units(retrieval_document: dict[str, Any], transcript_document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index retrieved units, enriching them with original transcript metadata."""
    indexed: dict[str, dict[str, Any]] = {}
    transcript_units = {
        unit.get("window_id") or unit.get("unit_id"): unit
        for unit in transcript_document.get("windows", [])
        if unit.get("window_id") or unit.get("unit_id")
    }
    transcript_units.update({
        unit.get("unit_id"): unit
        for unit in transcript_document.get("exchanges", [])
        if unit.get("unit_id")
    })
    for item in retrieval_document.get("results", []):
        unit_id = item.get("unit_id")
        if not unit_id:
            continue
        original = transcript_units.get(unit_id, {})
        indexed[unit_id] = {
            **item,
            "segment_ids": item.get("segment_ids") or original.get("segment_ids", []),
            "text": item.get("unit_text") or original.get("text", ""),
            "unit_type": item.get("unit_type") or original.get("unit_type", "window"),
        }
    return indexed


def get_checkpoint_context(
    checkpoint_id: str,
    retrieval_document: dict[str, Any],
    unit_index: dict[str, dict[str, Any]],
    max_units: int = MAX_UNITS,
) -> tuple[str, list[dict[str, Any]]]:
    """Return retrieval status and tagged units, falling back only when needed."""
    entry = next(
        (item for item in retrieval_document.get("by_checkpoint", []) if item.get("checkpoint_id") == checkpoint_id),
        None,
    )
    if entry is None:
        return "none", []
    tagged = entry.get("tagged_units") or []
    fallback = entry.get("fallback_units") or []
    selected = tagged or fallback
    units = [unit_index[item["unit_id"]] for item in selected if item.get("unit_id") in unit_index]
    units.sort(key=lambda unit: timestamp_key(unit.get("timestamp")))
    status = entry.get("retrieval_status") or "none"
    if not tagged and fallback:
        status = "fallback_only"
    return status, units[:max_units]


def build_prompt(checkpoint: dict[str, Any], units: list[dict[str, Any]]) -> str:
    """Build a source-anchored prompt for one checkpoint."""
    sources = []
    for unit in units:
        sources.append(
            f"SOURCE UNIT: {unit['unit_id']}\n"
            f"SEGMENTS: {', '.join(unit.get('segment_ids', []))}\n"
            f"TEXT:\n{unit.get('text', '')}"
        )
    return (
        f"CHECKPOINT ID: {checkpoint['checkpoint_id']}\n"
        f"PHASE: {checkpoint.get('phase', '')}\n"
        f"ACTIVITY: {checkpoint.get('activity', '')}\n"
        f"REQUIREMENT: {checkpoint.get('checkpoint_text', '')}\n\n"
        "SOURCE TRANSCRIPT UNITS:\n" + "\n\n".join(sources)
    )


def parse_llm_response(response_text: str) -> EvidenceResult:
    """Parse an LLM JSON response without allowing response decisions."""
    response_text = response_text.strip()
    if response_text.startswith("```"):
        response_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response_text, flags=re.IGNORECASE).strip()
    try:
        if hasattr(EvidenceResult, "model_validate_json"):
            return EvidenceResult.model_validate_json(response_text)
        return EvidenceResult.parse_raw(response_text)
    except (ValidationError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid evidence response: {exc}") from exc


def _quote_has_answer_content(quote: str) -> bool:
    """Reject support/contradiction based on an interviewer question alone."""
    lines = [line.strip() for line in quote.splitlines() if line.strip()]
    return bool(lines) and any(not line.endswith("?") for line in lines)


def validate_evidence_quotes(result: EvidenceResult, source_units: list[dict[str, Any]]) -> EvidenceResult:
    """Drop unsupported evidence and apply conservative review rules locally."""
    units_by_id = {unit["unit_id"]: unit for unit in source_units}
    valid_items: list[EvidenceItem] = []
    dropped: list[str] = []
    for item in result.items:
        source = units_by_id.get(item.unit_id)
        if source is None or not quote_matches(item.verbatim_quote, source.get("text", "")):
            dropped.append(f"Quote for {item.unit_id} did not match its source unit.")
            continue
        source_segments = set(source.get("segment_ids", []))
        if not set(item.segment_ids).issubset(source_segments):
            dropped.append(f"Segment IDs for {item.unit_id} were not present in the source unit.")
            continue
        if item.stance in {"supports", "contradicts"} and not _quote_has_answer_content(item.verbatim_quote):
            dropped.append(f"Question-only quote for {item.unit_id} was rejected.")
            continue
        valid_items.append(item)

    result.items = valid_items
    stances = {item.stance for item in valid_items}
    if {"supports", "contradicts"}.issubset(stances) and "supersedes" not in stances:
        result.evidence_status = "CONFLICTING"
        result.needs_review = True
        result.review_reason = "Supporting and contradicting evidence were both found without supersession."
    if result.confidence < MIN_CONFIDENCE:
        result.needs_review = True
        result.review_reason = result.review_reason or "Confidence is below the review threshold."
    if not valid_items:
        result.evidence_status = "NO_EVIDENCE"
        result.needs_review = True
        result.review_reason = result.review_reason or "No transcript quote passed local validation."
    elif result.evidence_status == "NO_EVIDENCE":
        result.evidence_status = "AMBIGUOUS"
        result.needs_review = True
        result.review_reason = result.review_reason or "The model returned evidence but marked the checkpoint as no evidence."
    if dropped:
        result.needs_review = True
        result.review_reason = result.review_reason or " ".join(dropped)
    return result


def _as_dict(result: EvidenceResult) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return result.dict()


def request_json(client: Any, model: str, messages: list[dict[str, str]]) -> str:
    """Request JSON, retrying without Groq JSON mode for incompatible models."""
    try:
        response = client.chat.completions.create(
            model=model, temperature=0, max_tokens=1200,
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
        model=model, temperature=0, max_tokens=1200, messages=messages,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Groq returned an empty evidence response")
    return content


def extract_evidence(
    registry_path: Path = REGISTRY_PATH,
    retrieval_path: Path = RETRIEVAL_PATH,
    transcript_path: Path = TRANSCRIPT_PATH,
    output_path: Path = OUTPUT_PATH,
    model: str | None = MODEL,
    limit: int | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Extract and validate evidence for every checkpoint."""
    registry = load_json(registry_path)
    retrieval = load_json(retrieval_path)
    transcript = load_json(transcript_path)
    checkpoints = registry.get("checkpoints", registry)
    unit_index = index_units(retrieval, transcript)
    injected_client = client is not None
    if client is None:
        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is required for evidence extraction")
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError("Install Groq with: python -m pip install groq") from exc
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
    if not injected_client:
        model = resolve_model(client, model)
    model = model or "injected-test-model"
    selected = checkpoints[:limit] if limit is not None else checkpoints
    records = []
    for checkpoint in selected:
        retrieval_status, units = get_checkpoint_context(checkpoint["checkpoint_id"], retrieval, unit_index)
        if not units:
            result = EvidenceResult(
                evidence_status="NO_EVIDENCE", items=[], observation="No retrieved transcript units were available.",
                confidence=0.0, needs_review=True, review_reason="No retrieved units were available."
            )
        else:
            content = request_json(
                client, model,
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_prompt(checkpoint, units)}],
            )
            result = validate_evidence_quotes(parse_llm_response(content), units)
        records.append({"checkpoint_id": checkpoint["checkpoint_id"], "retrieval_status": retrieval_status, **_as_dict(result)})
    payload = {
        "schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model, "prompt_hash": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "checkpoint_count": len(records), "evidence": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    """Run evidence extraction from retrieval results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    document = extract_evidence(model=args.model, limit=args.limit)
    print(f"Wrote {document['checkpoint_count']} evidence records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()