from pathlib import Path
import json
import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "checkpoint_registry.json"
)

TRANSCRIPT_PATH = (
    PROJECT_ROOT
    / "data"
    / "parsed_transcript.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "retrieval_results.json"
)


# =========================================================
# RETRIEVAL CONFIGURATION
# =========================================================

TOP_K = 5

# Weight given to semantic similarity
SEMANTIC_WEIGHT = 0.70

# Weight given to keyword overlap
KEYWORD_WEIGHT = 0.30

# Minimum combined score to keep a candidate
MIN_SCORE = 0.05


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text):
    """
    Normalize text for keyword matching.

    We intentionally keep the original transcript untouched.
    This normalized version is only used internally for retrieval.
    """

    if not text:
        return ""

    text = text.lower()

    # Keep letters, numbers and spaces.
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text):
    """
    Convert text into a set of normalized tokens.
    """

    normalized = normalize_text(text)

    if not normalized:
        return set()

    return set(normalized.split())


# =========================================================
# LOAD DATA
# =========================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def load_registry():
    print("\nLoading checkpoint registry...")

    registry = load_json(REGISTRY_PATH)

    # -----------------------------------------------------
    # Handle dictionary-based registry
    # -----------------------------------------------------

    if isinstance(registry, dict):

        # Case 1:
        # {
        #     "C001": {...},
        #     "C002": {...}
        # }
        #
        # Convert dictionary values into a list.

        if all(
            isinstance(value, dict)
            for value in registry.values()
        ):

            checkpoints = []

            for checkpoint_id, checkpoint in registry.items():

                checkpoint = checkpoint.copy()

                # Add checkpoint_id if it is not already present
                checkpoint.setdefault(
                    "checkpoint_id",
                    checkpoint_id
                )

                checkpoints.append(
                    checkpoint
                )

        # Case 2:
        # {
        #     "checkpoints": [...]
        # }
        elif "checkpoints" in registry:

            checkpoints = registry["checkpoints"]

        else:

            raise ValueError(
                "Unsupported checkpoint registry format."
            )

    # -----------------------------------------------------
    # Handle list-based registry
    # -----------------------------------------------------

    elif isinstance(registry, list):

        checkpoints = registry

    else:

        raise ValueError(
            "Checkpoint registry must be a list or dictionary."
        )

    # -----------------------------------------------------
    # Validate entries
    # -----------------------------------------------------

    valid_checkpoints = []

    for checkpoint in checkpoints:

        if not isinstance(checkpoint, dict):
            continue

        if not checkpoint.get("checkpoint_id"):
            continue

        valid_checkpoints.append(
            checkpoint
        )

    print(
        f"Loaded {len(valid_checkpoints)} checkpoints."
    )

    return valid_checkpoints


def load_transcript():
    print("\nLoading parsed transcript...")

    transcript = load_json(TRANSCRIPT_PATH)

    windows = transcript.get("windows", [])

    print(
        f"Loaded {len(windows)} conversational windows."
    )

    return transcript


# =========================================================
# CHECKPOINT TEXT PREPARATION
# =========================================================

def checkpoint_to_text(checkpoint):
    """
    Convert a checkpoint registry entry into searchable text.

    We include:
    - phase
    - activity
    - actual checkpoint text
    """

    parts = []

    if checkpoint.get("phase"):
        parts.append(
            str(checkpoint["phase"])
        )

    if checkpoint.get("activity"):
        parts.append(
            str(checkpoint["activity"])
        )

    if checkpoint.get("checkpoint_text"):
        parts.append(
            str(checkpoint["checkpoint_text"])
        )

    return " ".join(parts)


# =========================================================
# KEYWORD SIMILARITY
# =========================================================

def keyword_similarity(
    query_text,
    checkpoint_text
):
    """
    Calculate simple token-overlap similarity.

    This is not the final semantic understanding.
    It provides an additional retrieval signal.
    """

    query_tokens = tokenize(query_text)
    checkpoint_tokens = tokenize(checkpoint_text)

    if not query_tokens or not checkpoint_tokens:
        return 0.0

    overlap = query_tokens.intersection(
        checkpoint_tokens
    )

    # Jaccard similarity
    union = query_tokens.union(
        checkpoint_tokens
    )

    if not union:
        return 0.0

    return len(overlap) / len(union)


# =========================================================
# BUILD TF-IDF INDEX
# =========================================================

def build_tfidf_index(checkpoints):
    """
    Build a TF-IDF representation of all checkpoints.
    """

    checkpoint_texts = [
        checkpoint_to_text(checkpoint)
        for checkpoint in checkpoints
    ]
    

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    checkpoint_vectors = vectorizer.fit_transform(
        checkpoint_texts
    )

    return (
        vectorizer,
        checkpoint_vectors
    )


# =========================================================
# RETRIEVE CHECKPOINTS
# =========================================================

def retrieve_for_window(
    window,
    checkpoints,
    vectorizer,
    checkpoint_vectors
):
    """
    Retrieve the most relevant checkpoint candidates
    for one conversational window.
    """

    query_text = window.get(
        "text",
        ""
    )

    if not query_text:
        return []

    # -----------------------------------------------------
    # Semantic similarity
    # -----------------------------------------------------

    query_vector = vectorizer.transform(
        [query_text]
    )

    semantic_scores = cosine_similarity(
        query_vector,
        checkpoint_vectors
    )[0]

    # -----------------------------------------------------
    # Combined scoring
    # -----------------------------------------------------

    candidates = []

    for index, checkpoint in enumerate(
        checkpoints
    ):

        checkpoint_text = checkpoint_to_text(
            checkpoint
        )

        semantic_score = float(
            semantic_scores[index]
        )

        keyword_score = keyword_similarity(
            query_text,
            checkpoint_text
        )

        combined_score = (
            SEMANTIC_WEIGHT * semantic_score
            +
            KEYWORD_WEIGHT * keyword_score
        )

        if combined_score < MIN_SCORE:
            continue

        candidates.append(
            {
                "checkpoint_id":
                    checkpoint.get("checkpoint_id"),

                "checkpoint":
                    checkpoint.get("checkpoint_text"),

                "phase":
                    checkpoint.get("phase"),

                "activity":
                    checkpoint.get("activity"),

                "semantic_score":
                    round(
                        semantic_score,
                        4
                    ),

                "keyword_score":
                    round(
                        keyword_score,
                        4
                    ),

                "combined_score":
                    round(
                        combined_score,
                        4
                    )
            }
        )

    # Highest score first
    candidates.sort(
        key=lambda x: x["combined_score"],
        reverse=True
    )

    return candidates[:TOP_K]


# =========================================================
# RETRIEVE ALL WINDOWS
# =========================================================

def retrieve_all_windows(
    transcript,
    checkpoints
):
    windows = transcript.get(
        "windows",
        []
    )

    print(
        "\nBuilding TF-IDF checkpoint index..."
    )

    vectorizer, checkpoint_vectors = (
        build_tfidf_index(
            checkpoints
        )
    )

    print(
        "Checkpoint index created."
    )

    results = []

    total_windows = len(windows)

    print(
        f"\nRetrieving checkpoints for "
        f"{total_windows} windows..."
    )

    for index, window in enumerate(
        windows,
        start=1
    ):

        candidates = retrieve_for_window(
            window,
            checkpoints,
            vectorizer,
            checkpoint_vectors
        )

        result = {
            "window_id":
                window.get("window_id"),

            "center_segment_id":
                window.get(
                    "center_segment_id"
                ),

            "segment_ids":
                window.get(
                    "segment_ids",
                    []
                ),

            "window_text":
                window.get(
                    "text",
                    ""
                ),

            "candidates":
                candidates
        }

        results.append(result)

        # Progress display
        if (
            index == 1
            or index % 25 == 0
            or index == total_windows
        ):
            print(
                f"Processed "
                f"{index}/{total_windows} windows"
            )

    return results


# =========================================================
# SAVE RESULTS
# =========================================================

def save_results(
    transcript,
    results
):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "retriever": {
            "top_k": TOP_K,
            "semantic_weight":
                SEMANTIC_WEIGHT,
            "keyword_weight":
                KEYWORD_WEIGHT,
            "minimum_score":
                MIN_SCORE
        },

        "transcript_metadata":
            transcript.get(
                "metadata",
                {}
            ),

        "total_windows":
            len(results),

        "results":
            results
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\nRetrieval results saved to:\n"
        f"{OUTPUT_PATH}"
    )


# =========================================================
# DISPLAY SAMPLE RESULTS
# =========================================================

def display_sample_results(results):

    print("\n" + "=" * 70)
    print("SAMPLE RETRIEVAL RESULTS")
    print("=" * 70)

    for result in results[:3]:

        print(
            f"\nWindow: "
            f"{result['window_id']}"
        )

        print(
            f"Center segment: "
            f"{result['center_segment_id']}"
        )

        print(
            "\nConversation:"
        )

        print(
            result["window_text"]
        )

        print(
            "\nTop checkpoint candidates:"
        )

        for rank, candidate in enumerate(
            result["candidates"],
            start=1
        ):

            print(
                f"\n{rank}. "
                f"{candidate['checkpoint_id']} "
                f"| Score: "
                f"{candidate['combined_score']}"
            )

            print(
                f"   Phase: "
                f"{candidate['phase']}"
            )

            print(
                f"   Activity: "
                f"{candidate['activity']}"
            )

            print(
                f"   Checkpoint: "
                f"{candidate['checkpoint']}"
            )

            print(
                f"   Semantic: "
                f"{candidate['semantic_score']} | "
                f"Keyword: "
                f"{candidate['keyword_score']}"
            )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("SEPG CHECKPOINT RETRIEVER")
    print("=" * 70)

    # -----------------------------------------------------
    # Load checkpoint registry
    # -----------------------------------------------------

    checkpoints = load_registry()

    # -----------------------------------------------------
    # Load parsed transcript
    # -----------------------------------------------------

    transcript = load_transcript()

    # -----------------------------------------------------
    # Retrieve candidates
    # -----------------------------------------------------

    results = retrieve_all_windows(
        transcript,
        checkpoints
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    save_results(
        transcript,
        results
    )

    # -----------------------------------------------------
    # Show samples
    # -----------------------------------------------------

    display_sample_results(
        results
    )

    print("\n" + "=" * 70)
    print(
        "Checkpoint retrieval completed successfully."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()