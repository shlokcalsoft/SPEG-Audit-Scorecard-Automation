# Change Log

## 2026-09-21

- Files added: `src/evidence_extractor.py`, `tests/test_evidence_extractor.py`, `requirements.txt`, `CHANGES.md`.
- Files modified: none.
- What changed and why: Implemented step 4 as a standalone evidence-only stage. It resolves retrieved units per checkpoint, uses fallback units only when tagged units are absent, sends at most eight timestamp-ordered units to Groq, validates exact transcript-backed quotes locally, and never emits Yes/No/Partial/NA decisions.
- JSON contracts/config keys: Added `data/evidence_results.json` with `schema_version`, `generated_at`, `model`, `prompt_hash`, `checkpoint_count`, and `evidence`. Each record contains `checkpoint_id`, `retrieval_status`, `evidence_status`, `items`, `observation`, `confidence`, `needs_review`, and `review_reason`. Each evidence item contains `unit_id`, `segment_ids`, `timestamp`, `speaker`, `verbatim_quote`, `stance`, and `reason`.
- How to run or test: `py -m pip install -r requirements.txt`; run `py -m pytest tests/test_evidence_extractor.py -q`; set `GROQ_API_KEY`; run `py -m src.evidence_extractor --limit 3`.
- Known issues / TODOs: Live Groq extraction was not run because no API key is configured. The current retriever output uses windows, which the extractor supports; exchange artifacts can be added later without changing this contract.

## 2026-09-21

- Files added: `src/assessment_engine.py`, `tests/test_assessment_engine.py`.
- Files modified: `CHANGES.md`.
- What changed and why: Implemented step 5. The engine consumes validated evidence only, maps evidence to Yes/Partial/No/NA/Not Clarified, forces review for ambiguous/conflicting evidence and confidence below 0.6, and deterministically maps missing evidence to Not Clarified without an LLM call.
- JSON contracts/config keys: Added `data/assessments.json` with `schema_version`, `generated_at`, `model`, `prompt_hash`, `checkpoint_count`, and `assessments`; each assessment contains `checkpoint_id`, `response`, `observation`, `confidence`, `needs_review`, `review_reason`, `evidence_status`, and validated evidence.
- How to run or test: `py -m pytest tests/test_assessment_engine.py -q`; set `GROQ_API_KEY`; run `py -m src.assessment_engine --limit 3`.
- Known issues / TODOs: Live assessment requires Groq access for evidence-bearing checkpoints. Conflict resolution remains conservative and is intentionally left flagged for step 6.

## 2026-09-21

- Files added: `.env.example`, `.gitignore`.
- Files modified: `src/evidence_extractor.py`, `src/assessment_engine.py`, `requirements.txt`, `CHANGES.md`.
- What changed and why: Added safe local `.env` loading for `GROQ_API_KEY` using `python-dotenv`; the real `.env` remains ignored and was never read or exposed.
- JSON contracts/config keys: Added environment key `GROQ_API_KEY`; no JSON contract changes.
- How to run or test: `py -m pip install -r requirements.txt`; run `py -m pytest tests/test_evidence_extractor.py tests/test_assessment_engine.py -q`; then run the live commands below.
- Known issues / TODOs: Do not commit `.env`. Live Groq calls require a valid key and may incur API usage.

## 2026-09-22

- Files added: none.
- Files modified: `src/evidence_extractor.py`, `src/assessment_engine.py`, `.env.example`, `CHANGES.md`.
- What changed and why: Replaced the unavailable `llama-3.3-70b-versatile` default with Groq's free-tier-compatible `llama-3.1-8b-instant`; added optional `GROQ_MODEL` configuration.
- JSON contracts/config keys: Added optional environment key `GROQ_MODEL`; output contracts unchanged.
- How to run or test: `py -m src.evidence_extractor --limit 1`; then `py -m src.assessment_engine --limit 1`.
- Known issues / TODOs: Model availability is controlled by Groq and can change; set `GROQ_MODEL` to another model available to your account if needed.

## 2026-09-22

- Files added: none.
- Files modified: `src/evidence_extractor.py`, `src/assessment_engine.py`, `.env.example`, `CHANGES.md`.
- What changed and why: Added Groq model discovery. If `GROQ_MODEL` is unset or unavailable, each stage lists account-accessible models and selects a preferred text model automatically.
- JSON contracts/config keys: `GROQ_MODEL` remains optional; output model metadata now records the resolved model.
- How to run or test: `py -m src.evidence_extractor --limit 1`; inspect `data/evidence_results.json` for the resolved model.
- Known issues / TODOs: Model discovery requires the key to have permission to list models; explicitly set `GROQ_MODEL` when deterministic model selection is required.

## 2026-09-22

- Files added: `data/evidence_results.json`, `data/assessments.json` (local generated artifacts; ignored by git).
- Files modified: `src/assessment_engine.py`, `CHANGES.md`.
- What changed and why: Removed a redundant model-resolution call from the no-evidence assessment path and verified live execution with the account-accessible `openai/gpt-oss-20b` model.
- JSON contracts/config keys: No contract changes; resolved model is recorded in the generated evidence and assessment metadata.
- How to run or test: `py -m pytest tests/test_evidence_extractor.py tests/test_assessment_engine.py -q` (12 passed); `py -m src.evidence_extractor --limit 1`; `py -m src.assessment_engine --limit 1`.
- Known issues / TODOs: Full 248-checkpoint execution will make one Groq request per evidence-bearing checkpoint and may consume free-tier quota.

## 2026-09-22

- Files added: none.
- Files modified: `src/evidence_extractor.py`, `src/assessment_engine.py`, `CHANGES.md`.
- What changed and why: Added a compatibility retry for Groq models that reject JSON response mode. The stages first request structured JSON, then retry without `response_format` and parse fenced JSON when necessary.
- JSON contracts/config keys: No contract changes; evidence and assessment output schemas are unchanged.
- How to run or test: `py -m pytest tests/test_evidence_extractor.py tests/test_assessment_engine.py -q` (12 passed); live `py -m src.evidence_extractor --limit 1` and `py -m src.assessment_engine --limit 1` both completed successfully.
- Known issues / TODOs: Full runs may consume Groq free-tier quota and take time for all checkpoints.

## 2026-09-21

- Files added: none.
- Files modified: `src/assessment_engine.py`, `CHANGES.md`.
- What changed and why: Exposed the public `parse_assessment()` helper used by callers and tests, while retaining the same Pydantic validation path.
- JSON contracts/config keys: none.
- How to run or test: `py -m pytest tests/test_evidence_extractor.py tests/test_assessment_engine.py -q` (12 passed).
- Known issues / TODOs: `data/evidence_results.json` must exist before the live assessment CLI can run; it is produced by step 4.

## 2026-09-21

- Files added: `.env.example`, `.gitignore`.
- Files modified: `src/evidence_extractor.py`, `src/assessment_engine.py`, `requirements.txt`, `CHANGES.md`.
- What changed and why: Added local `.env` loading for `GROQ_API_KEY` using `python-dotenv`, while ignoring the real secret and generated assessment artifacts from git.
- JSON contracts/config keys: Added environment key `GROQ_API_KEY`; no JSON contract changes.
- How to run or test: Copy `.env.example` to `.env`, set the key, install requirements, then run the step 4 and 5 commands below.
- Known issues / TODOs: Never commit `.env`; rotate the key immediately if it is accidentally exposed.