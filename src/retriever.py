"""
SEPG checkpoint retriever (hybrid).

Job: look at a piece of the transcript (an exchange or a window) and
find which SEPG checkpoints it might be evidence for.

How it works
------------
1. Clean the unit text (speaker/timestamp prefixes, filler words).
2. Score every unit against every checkpoint with two arms:
     - lexical : BM25 (ranking) + IDF-weighted containment (cut-off)
     - dense   : sentence-embedding cosine similarity (optional)
   Each arm is one matrix operation over ALL units x ALL checkpoints.
3. Fuse the two rankings with Reciprocal Rank Fusion (RRF).
4. Add a small SOFT bonus for checkpoints whose Activity matches the unit
   (never a hard filter).
5. Apply cut-offs: a candidate is kept only if it passes an absolute
   relevance gate (dense cosine OR lexical containment). Units with no
   passing candidate return an empty list ("nothing relevant here").
6. Read the same score matrix column-wise for checkpoints no unit was
   tagged with (fallback), so "not discussed" is a checked result.

This module only finds candidate evidence. It never decides Yes/No/Partial.
`retrieval_status` describes retrieval only, not the audit answer.

Optional inputs
---------------
data/checkpoint_hints.json  {"C024": {"aliases": [...], "typical_questions": [...]}}
transcript["exchanges"]     preferred over transcript["windows"] when present
"""

from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import re
import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REGISTRY_PATH = PROJECT_ROOT / "data" / "checkpoint_registry.json"
TRANSCRIPT_PATH = PROJECT_ROOT / "data" / "parsed_transcript.json"
HINTS_PATH = PROJECT_ROOT / "data" / "checkpoint_hints.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "retrieval_results.json"
CACHE_DIR = PROJECT_ROOT / "data" / ".cache"


# =========================================================
# CONFIGURATION
# =========================================================
# NOTE: min_dense and min_containment are PLACEHOLDERS. They depend on the
# embedding model and must be calibrated on your gold-labelled units
# (see the calibration report printed at the end of every run).

CONFIG = {
    # Output sizes
    "top_k": 5,              # candidates kept per unit
    "fallback_k": 5,         # units kept per checkpoint nobody tagged
    "evidence_k": 8,         # max tagged units listed per checkpoint

    # Fusion
    "rrf_k": 60,
    "activity_weight": 0.25,  # soft activity bonus (0 disables it)

    # Absolute relevance gate: dense cosine OR lexical containment
    "min_dense": 0.30,
    "min_containment": 0.30,

    # BM25
    "bm25_k1": 1.5,
    "bm25_b": 0.75,

    # Dense arm
    "use_dense": True,
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",

    # Whether hint text is also added to the lexical arm (test on gold set)
    "hints_in_lexical": False,
}


# =========================================================
# TEXT CLEANING AND TOKENIZATION
# =========================================================

# "Amit Gogate [0:16]:" at the start of a line
SPEAKER_PREFIX_RE = re.compile(
    r"^[^\n\[\]:]{1,60}\[\d{1,2}:\d{2}(?::\d{2})?\]\s*:?\s*",
    re.MULTILINE
)

# any remaining "[0:16]" timestamps
TIMESTAMP_RE = re.compile(r"\[\d{1,2}:\d{2}(?::\d{2})?\]")

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small hand-written list. (scikit-learn's list contains words such as
# "system", which are meaningful in audit language.)
STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been
before being below between both but by can could did do does doing down
during each few for from further had has have having he her here hers him
his how i if in into is it its just me more most my no nor not of off on
once only or other our out over own same she should so some such than that
the their them then there these they this those through to too under until
up us very was we were what when where which while who whom why will with
would you your
yeah yes yep okay ok mhm hmm um uh right actually basically like sort kind
thing things know think mean means see sure thanks thank please hello hi
""".split())


def clean_unit_text(text):
    """
    Remove speaker/timestamp prefixes and collapse whitespace.
    The original transcript is never modified; this is retrieval-only.
    """

    if not text:
        return ""

    text = SPEAKER_PREFIX_RE.sub(" ", text)
    text = TIMESTAMP_RE.sub(" ", text)

    return re.sub(r"\s+", " ", text).strip()


def light_stem(token):
    """
    Plural normalisation only (tests -> test, policies -> policy).
    Deliberately NOT a full stemmer, so "developer" and "development"
    stay different words.
    """

    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"

    if (
        len(token) > 3
        and token.endswith("s")
        and not token.endswith(("ss", "us", "is"))
    ):
        return token[:-1]

    return token


def lexical_tokens(text):
    return [
        light_stem(token)
        for token in TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


# =========================================================
# LOAD DATA
# =========================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_registry():
    print("\nLoading checkpoint registry...")

    registry = load_json(REGISTRY_PATH)

    if isinstance(registry, list):
        checkpoints = registry

    elif isinstance(registry, dict):

        # {"C001": {...}, "C002": {...}}
        if all(isinstance(v, dict) for v in registry.values()):
            checkpoints = []
            for checkpoint_id, checkpoint in registry.items():
                checkpoint = checkpoint.copy()
                checkpoint.setdefault("checkpoint_id", checkpoint_id)
                checkpoints.append(checkpoint)

        # {"checkpoints": [...]}
        elif "checkpoints" in registry:
            checkpoints = registry["checkpoints"]

        else:
            raise ValueError("Unsupported checkpoint registry format.")

    else:
        raise ValueError("Checkpoint registry must be a list or dictionary.")

    valid = [
        checkpoint
        for checkpoint in checkpoints
        if isinstance(checkpoint, dict) and checkpoint.get("checkpoint_id")
    ]

    print(f"Loaded {len(valid)} checkpoints.")

    return valid


def load_hints():
    if not HINTS_PATH.exists():
        return {}

    hints = load_json(HINTS_PATH)

    if not isinstance(hints, dict):
        raise ValueError("checkpoint_hints.json must map checkpoint_id -> hint.")

    print(f"Loaded hints for {len(hints)} checkpoints.")

    return hints


def load_transcript():
    print("\nLoading parsed transcript...")

    transcript = load_json(TRANSCRIPT_PATH)

    return transcript


def load_units(transcript):
    """
    Retrieval units: Q&A exchanges if the parser produced them,
    otherwise the conversational windows.
    """

    units = []

    exchanges = transcript.get("exchanges") or []

    if exchanges:
        for exchange in exchanges:
            text = exchange.get("text") or " ".join(
                part
                for part in (exchange.get("question"), exchange.get("answer"))
                if part
            )

            units.append(
                {
                    "unit_id": exchange.get("unit_id"),
                    "unit_type": "exchange",
                    "center_segment_id": None,
                    "segment_ids": exchange.get("segment_ids", []),
                    "text": text or "",
                }
            )

    else:
        for window in transcript.get("windows", []):
            units.append(
                {
                    "unit_id": window.get("window_id"),
                    "unit_type": "window",
                    "center_segment_id": window.get("center_segment_id"),
                    "segment_ids": window.get("segment_ids", []),
                    "text": window.get("text") or "",
                }
            )

    print(f"Loaded {len(units)} {units[0]['unit_type'] + 's' if units else 'units'}.")

    return units


# =========================================================
# CHECKPOINT TEXT
# =========================================================

def hint_text(hint):
    if not hint:
        return ""

    parts = []
    parts.extend(hint.get("aliases", []))
    parts.extend(hint.get("typical_questions", []))

    return " ".join(str(part) for part in parts)


def checkpoint_base_text(checkpoint):
    return str(checkpoint.get("checkpoint_text") or "")


def checkpoint_lexical_text(checkpoint, hint=None):
    parts = [
        str(checkpoint.get("phase") or ""),
        str(checkpoint.get("activity") or ""),
        checkpoint_base_text(checkpoint),
    ]

    if CONFIG["hints_in_lexical"]:
        parts.append(hint_text(hint))

    return " ".join(part for part in parts if part)


def checkpoint_dense_text(checkpoint, hint=None):
    text = (
        f"{checkpoint.get('phase') or ''} - "
        f"{checkpoint.get('activity') or ''}: "
        f"{checkpoint_base_text(checkpoint)}."
    )

    extra = hint_text(hint)

    if extra:
        text += f" {extra}"

    return text


# =========================================================
# LEXICAL ARM: BM25 + IDF-WEIGHTED CONTAINMENT
# =========================================================

def build_lexical_index(doc_texts):
    """
    Index a list of documents (checkpoints or activities).

    Returns BM25 term weights and IDF-weighted term presence, so scoring
    all units is a single sparse matrix product.
    """

    vectorizer = CountVectorizer(analyzer=lexical_tokens)

    tf = vectorizer.fit_transform(doc_texts).tocsr().astype(np.float64)

    n_docs = tf.shape[0]

    doc_freq = np.bincount(tf.indices, minlength=tf.shape[1])

    idf = np.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

    doc_len = np.asarray(tf.sum(axis=1)).ravel()
    avg_len = doc_len.mean() if doc_len.mean() > 0 else 1.0

    rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))

    k1 = CONFIG["bm25_k1"]
    b = CONFIG["bm25_b"]

    denominator = tf.data + k1 * (1.0 - b + b * doc_len[rows] / avg_len)

    weights = tf.copy()
    weights.data = tf.data * (k1 + 1.0) / denominator * idf[tf.indices]

    presence = tf.copy()
    presence.data = idf[tf.indices]

    idf_mass = np.asarray(presence.sum(axis=1)).ravel()

    return {
        "vectorizer": vectorizer,
        "weights": weights,
        "presence": presence,
        "idf_mass": idf_mass,
    }


def lexical_scores(index, unit_texts):
    """
    Returns (bm25, containment), each shaped (n_units, n_docs).

    bm25         ranking signal (scale depends on the unit, so it is
                 used for ranking only)
    containment  share of the document's IDF mass found in the unit,
                 in [0, 1]. An absolute, interpretable cut-off signal that
                 is not diluted by long units (unlike Jaccard).
    """

    query = index["vectorizer"].transform(unit_texts).tocsr().astype(np.float64)
    query.data[:] = 1.0  # each query term counts once

    bm25 = (query @ index["weights"].T).toarray()

    overlap = (query @ index["presence"].T).toarray()

    containment = np.divide(
        overlap,
        index["idf_mass"],
        out=np.zeros_like(overlap),
        where=index["idf_mass"] > 0,
    )

    return bm25, containment


# =========================================================
# DENSE ARM: SENTENCE EMBEDDINGS (cached)
# =========================================================

_MODEL_CACHE = {}


def encode_with_model(model_name, texts):
    """
    Encode texts into L2-normalised vectors. Imported lazily so the
    project still runs (lexical-only) without sentence-transformers.
    """

    from sentence_transformers import SentenceTransformer

    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)

    vectors = _MODEL_CACHE[model_name].encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(vectors, dtype=np.float32)


def embed_texts(texts, model_name):
    """
    Embed texts, caching vectors on disk by content hash so unchanged
    checkpoints/units are never re-encoded.
    """

    slug = re.sub(r"[^A-Za-z0-9]+", "_", model_name)
    cache_path = CACHE_DIR / f"embeddings_{slug}.npz"

    cache = {}

    if cache_path.exists():
        with np.load(cache_path) as stored:
            cache = {key: stored[key] for key in stored.files}

    keys = [
        hashlib.sha1(text.encode("utf-8")).hexdigest()
        for text in texts
    ]

    missing = {}

    for key, text in zip(keys, texts):
        if key not in cache and key not in missing:
            missing[key] = text

    if missing:
        vectors = encode_with_model(model_name, list(missing.values()))

        for key, vector in zip(missing, vectors):
            cache[key] = vector

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **cache)

    return np.vstack([cache[key] for key in keys])


def activity_centroids(checkpoint_vectors, activity_index, n_activities):
    """
    Mean checkpoint embedding per activity (no extra model calls).
    """

    centroids = np.zeros(
        (n_activities, checkpoint_vectors.shape[1]),
        dtype=np.float32
    )

    np.add.at(centroids, activity_index, checkpoint_vectors)

    norms = np.linalg.norm(centroids, axis=1, keepdims=True)

    return centroids / np.where(norms == 0, 1.0, norms)


# =========================================================
# FUSION
# =========================================================

def rank_matrix(scores):
    """1-based rank of every column within each row (1 = best)."""

    order = np.argsort(-scores, axis=1, kind="stable")

    ranks = np.empty_like(order)

    np.put_along_axis(
        ranks,
        order,
        np.broadcast_to(
            np.arange(1, scores.shape[1] + 1),
            scores.shape
        ),
        axis=1,
    )

    return ranks


def rrf_fuse(arms, rrf_k):
    """
    Reciprocal Rank Fusion, normalised to (0, 1].

    arms: list of (score_matrix, positive_only)
    positive_only=True gives zero credit to zero-score items, so a unit
    with no lexical overlap does not receive arbitrary tie-break credit.
    """

    total = np.zeros(arms[0][0].shape, dtype=np.float64)

    for scores, positive_only in arms:

        contribution = 1.0 / (rrf_k + rank_matrix(scores))

        if positive_only:
            contribution = np.where(scores > 0, contribution, 0.0)

        total += contribution

    return total / (len(arms) / (rrf_k + 1.0))


# =========================================================
# SCORING PIPELINE
# =========================================================

def compute_scores(unit_texts, checkpoints, hints, use_dense=True):
    """
    Score every unit against every checkpoint.

    Returns a dict of (n_units, n_checkpoints) matrices plus flags.
    """

    n_checkpoints = len(checkpoints)

    # -----------------------------------------------------
    # Activity grouping (Phase, Activity)
    # -----------------------------------------------------

    activity_lookup = {}
    activity_members = defaultdict(list)
    activity_index = np.empty(n_checkpoints, dtype=np.int64)

    for position, checkpoint in enumerate(checkpoints):

        key = (
            str(checkpoint.get("phase") or ""),
            str(checkpoint.get("activity") or ""),
        )

        if key not in activity_lookup:
            activity_lookup[key] = len(activity_lookup)

        activity_index[position] = activity_lookup[key]
        activity_members[activity_lookup[key]].append(position)

    n_activities = len(activity_lookup)

    activity_texts = []

    for (phase, activity), number in activity_lookup.items():
        members = " ".join(
            checkpoint_base_text(checkpoints[i])
            for i in activity_members[number]
        )
        activity_texts.append(f"{phase} {activity} {members}")

    # -----------------------------------------------------
    # Lexical arm (checkpoints and activities)
    # -----------------------------------------------------

    checkpoint_lex_texts = [
        checkpoint_lexical_text(
            checkpoint,
            hints.get(checkpoint.get("checkpoint_id"))
        )
        for checkpoint in checkpoints
    ]

    bm25, containment = lexical_scores(
        build_lexical_index(checkpoint_lex_texts),
        unit_texts
    )

    activity_bm25, _ = lexical_scores(
        build_lexical_index(activity_texts),
        unit_texts
    )

    # -----------------------------------------------------
    # Dense arm (optional)
    # -----------------------------------------------------

    dense = None
    activity_dense = None

    if use_dense:

        try:
            checkpoint_dense_texts = [
                checkpoint_dense_text(
                    checkpoint,
                    hints.get(checkpoint.get("checkpoint_id"))
                )
                for checkpoint in checkpoints
            ]

            model_name = CONFIG["model_name"]

            unit_vectors = embed_texts(unit_texts, model_name)
            checkpoint_vectors = embed_texts(checkpoint_dense_texts, model_name)

            dense = unit_vectors @ checkpoint_vectors.T

            activity_dense = unit_vectors @ activity_centroids(
                checkpoint_vectors,
                activity_index,
                n_activities
            ).T

        except Exception as error:
            print(
                f"\nWARNING: dense retrieval unavailable "
                f"({type(error).__name__}: {error}).\n"
                f"Continuing with the lexical arm only."
            )
            dense = None
            activity_dense = None

    # -----------------------------------------------------
    # Fusion: checkpoint level
    # -----------------------------------------------------

    arms = [(bm25, True)]

    if dense is not None:
        arms.append((dense, False))

    fused = rrf_fuse(arms, CONFIG["rrf_k"])

    # -----------------------------------------------------
    # Fusion: activity level (soft prior in [0, 1])
    # -----------------------------------------------------

    activity_arms = [(activity_bm25, True)]

    if activity_dense is not None:
        activity_arms.append((activity_dense, False))

    activity_prior = rrf_fuse(activity_arms, CONFIG["rrf_k"])

    final = fused + CONFIG["activity_weight"] * activity_prior[:, activity_index]

    # -----------------------------------------------------
    # Absolute relevance gate
    # -----------------------------------------------------

    lexical_pass = containment >= CONFIG["min_containment"]

    if dense is not None:
        dense_pass = dense >= CONFIG["min_dense"]
    else:
        dense_pass = np.zeros_like(lexical_pass)

    nonempty = np.array([bool(text) for text in unit_texts])

    gate = (lexical_pass | dense_pass) & nonempty[:, None]

    return {
        "bm25": bm25,
        "containment": containment,
        "dense": dense,
        "fused": fused,
        "activity_prior": activity_prior[:, activity_index],
        "final": final,
        "lexical_pass": lexical_pass,
        "dense_pass": dense_pass,
        "gate": gate,
        "nonempty": nonempty,
    }


def select_top_k(final, gate, top_k):
    """
    Per unit: indices of the best top_k checkpoints among those passing
    the gate. Returns (indices, valid_mask). Ties keep registry order.
    """

    ranked = np.where(gate, final, -np.inf)

    top_indices = np.argsort(-ranked, axis=1, kind="stable")[:, :top_k]

    valid = np.take_along_axis(ranked, top_indices, axis=1) > -np.inf

    return top_indices, valid


# =========================================================
# BUILD OUTPUT
# =========================================================

def r4(value):
    return None if value is None else round(float(value), 4)


def build_unit_results(units, checkpoints, scores, top_indices, valid):

    results = []

    for row, unit in enumerate(units):

        candidates = []

        for column, is_valid in zip(top_indices[row], valid[row]):

            if not is_valid:
                continue

            checkpoint = checkpoints[column]

            signals = []

            if scores["dense_pass"][row, column]:
                signals.append("dense")

            if scores["lexical_pass"][row, column]:
                signals.append("lexical")

            dense_value = (
                None
                if scores["dense"] is None
                else scores["dense"][row, column]
            )

            candidates.append(
                {
                    "checkpoint_id": checkpoint.get("checkpoint_id"),
                    "checkpoint": checkpoint.get("checkpoint_text"),
                    "phase": checkpoint.get("phase"),
                    "activity": checkpoint.get("activity"),
                    "final_score": r4(scores["final"][row, column]),
                    "fused_score": r4(scores["fused"][row, column]),
                    "activity_prior": r4(scores["activity_prior"][row, column]),
                    "bm25_score": r4(scores["bm25"][row, column]),
                    "lexical_containment": r4(scores["containment"][row, column]),
                    "dense_score": r4(dense_value),
                    "signals": signals,
                }
            )

        results.append(
            {
                "unit_id": unit["unit_id"],
                "unit_type": unit["unit_type"],
                "center_segment_id": unit["center_segment_id"],
                "segment_ids": unit["segment_ids"],
                "unit_text": unit["text"],
                "retrieval_status": (
                    "candidates_found" if candidates else "none"
                ),
                "candidates": candidates,
            }
        )

    return results


def build_checkpoint_view(units, checkpoints, scores, top_indices, valid):
    """
    Group row-wise tags by checkpoint. For checkpoints nobody tagged,
    read the same matrix column-wise (fallback) so that "not discussed"
    is a checked result rather than retrieval silence.
    """

    final = scores["final"]
    gate = scores["gate"]

    tagged = defaultdict(list)

    for row in range(len(units)):
        for column, is_valid in zip(top_indices[row], valid[row]):
            if is_valid:
                tagged[column].append((row, final[row, column]))

    view = []

    for column, checkpoint in enumerate(checkpoints):

        entry = {
            "checkpoint_id": checkpoint.get("checkpoint_id"),
            "retrieval_status": None,
            "phase": checkpoint.get("phase"),
            "activity": checkpoint.get("activity"),
            "checkpoint": checkpoint.get("checkpoint_text"),
            "tagged_units": [],
            "fallback_units": [],
            "closest_unit": None,
        }

        rows = sorted(tagged[column], key=lambda item: -item[1])

        if rows:
            entry["retrieval_status"] = "tagged"
            entry["tagged_units"] = [
                {"unit_id": units[row]["unit_id"], "final_score": r4(score)}
                for row, score in rows[: CONFIG["evidence_k"]]
            ]

        else:
            column_scores = np.where(gate[:, column], final[:, column], -np.inf)

            order = np.argsort(-column_scores, kind="stable")[: CONFIG["fallback_k"]]

            entry["fallback_units"] = [
                {
                    "unit_id": units[row]["unit_id"],
                    "final_score": r4(column_scores[row]),
                }
                for row in order
                if column_scores[row] > -np.inf
            ]

            if entry["fallback_units"]:
                entry["retrieval_status"] = "fallback_only"

            else:
                entry["retrieval_status"] = "none"

                # Diagnostic only: best unit even though it failed the gate.
                if len(units):
                    best = int(np.argmax(final[:, column]))
                    entry["closest_unit"] = {
                        "unit_id": units[best]["unit_id"],
                        "final_score": r4(final[best, column]),
                    }

        view.append(entry)

    return view


# =========================================================
# ORCHESTRATION
# =========================================================

def retrieve(units, checkpoints, hints):

    unit_texts = [clean_unit_text(unit["text"]) for unit in units]

    print(
        f"\nScoring {len(units)} units against "
        f"{len(checkpoints)} checkpoints..."
    )

    started = time.perf_counter()

    scores = compute_scores(
        unit_texts,
        checkpoints,
        hints,
        use_dense=CONFIG["use_dense"]
    )

    top_indices, valid = select_top_k(
        scores["final"],
        scores["gate"],
        CONFIG["top_k"]
    )

    unit_results = build_unit_results(
        units, checkpoints, scores, top_indices, valid
    )

    checkpoint_view = build_checkpoint_view(
        units, checkpoints, scores, top_indices, valid
    )

    print(f"Scoring finished in {time.perf_counter() - started:.2f}s")

    return unit_results, checkpoint_view, scores


# =========================================================
# SAVE RESULTS
# =========================================================

def config_summary(scores, hints):

    summary = dict(CONFIG)

    summary["dense_used"] = scores["dense"] is not None
    summary["hints_loaded"] = len(hints)

    summary["config_hash"] = hashlib.sha1(
        json.dumps(summary, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    return summary


def save_results(transcript, unit_results, checkpoint_view, scores, hints):

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "retriever": config_summary(scores, hints),
        "transcript_metadata": transcript.get("metadata", {}),
        "total_units": len(unit_results),
        "units_with_candidates": sum(
            1 for r in unit_results if r["candidates"]
        ),
        "results": unit_results,
        "by_checkpoint": checkpoint_view,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"\nRetrieval results saved to:\n{OUTPUT_PATH}")


# =========================================================
# REPORTS
# =========================================================

def print_calibration_report(unit_results, checkpoint_view, scores):
    """
    Score distributions to help set min_dense / min_containment
    against your gold-labelled units.
    """

    nonempty = scores["nonempty"]

    def percentiles(values):
        if not len(values):
            return "n/a"
        return " ".join(
            f"p{p}={np.percentile(values, p):.2f}"
            for p in (10, 25, 50, 75, 90)
        )

    print("\n" + "=" * 70)
    print("CALIBRATION REPORT (best score per unit)")
    print("=" * 70)

    print(
        "containment: "
        + percentiles(scores["containment"].max(axis=1)[nonempty])
    )

    if scores["dense"] is not None:
        print(
            "dense cosine: "
            + percentiles(scores["dense"].max(axis=1)[nonempty])
        )

    empty_units = sum(1 for r in unit_results if not r["candidates"])

    print(f"\nUnits with no candidates: {empty_units}/{len(unit_results)}")

    for status in ("tagged", "fallback_only", "none"):
        count = sum(
            1 for c in checkpoint_view if c["retrieval_status"] == status
        )
        print(f"Checkpoints {status}: {count}/{len(checkpoint_view)}")

    print(
        "\nIf chit-chat units still return candidates, raise the gates; "
        "if clearly relevant units return none, lower them."
    )


def display_sample_results(unit_results):

    print("\n" + "=" * 70)
    print("SAMPLE RETRIEVAL RESULTS")
    print("=" * 70)

    for result in unit_results[:3]:

        print(f"\nUnit: {result['unit_id']} ({result['unit_type']})")
        print(f"Status: {result['retrieval_status']}")
        print("\nConversation:")
        print(result["unit_text"])

        for rank, candidate in enumerate(result["candidates"], start=1):

            print(
                f"\n{rank}. {candidate['checkpoint_id']} "
                f"| final {candidate['final_score']} "
                f"| signals {candidate['signals']}"
            )
            print(f"   {candidate['phase']} / {candidate['activity']}")
            print(f"   {candidate['checkpoint']}")
            print(
                f"   bm25 {candidate['bm25_score']} | "
                f"containment {candidate['lexical_containment']} | "
                f"dense {candidate['dense_score']} | "
                f"activity {candidate['activity_prior']}"
            )


# =========================================================
# MAIN
# =========================================================

def parse_args(argv=None):

    parser = argparse.ArgumentParser(
        description="SEPG checkpoint retriever (hybrid)."
    )

    parser.add_argument("--no-dense", action="store_true",
                        help="lexical arm only (no sentence-transformers)")
    parser.add_argument("--model", default=None,
                        help="sentence-transformers model name")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-dense", type=float, default=None)
    parser.add_argument("--min-containment", type=float, default=None)
    parser.add_argument("--activity-weight", type=float, default=None)

    return parser.parse_args(argv)


def apply_args(args):

    if args.no_dense:
        CONFIG["use_dense"] = False

    if args.model:
        CONFIG["model_name"] = args.model

    if args.top_k is not None:
        CONFIG["top_k"] = args.top_k

    if args.min_dense is not None:
        CONFIG["min_dense"] = args.min_dense

    if args.min_containment is not None:
        CONFIG["min_containment"] = args.min_containment

    if args.activity_weight is not None:
        CONFIG["activity_weight"] = args.activity_weight


def main(argv=None):

    apply_args(parse_args(argv))

    print("=" * 70)
    print("SEPG CHECKPOINT RETRIEVER (HYBRID)")
    print("=" * 70)

    checkpoints = load_registry()
    hints = load_hints()
    transcript = load_transcript()
    units = load_units(transcript)

    unit_results, checkpoint_view, scores = retrieve(
        units, checkpoints, hints
    )

    save_results(transcript, unit_results, checkpoint_view, scores, hints)
    display_sample_results(unit_results)
    print_calibration_report(unit_results, checkpoint_view, scores)

    print("\n" + "=" * 70)
    print("Checkpoint retrieval completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
