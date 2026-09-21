from pathlib import Path
import json
import re

from docx import Document

from pathlib import Path

TRANSCRIPT_DIR = Path("data/transcripts")

docx_files = list(TRANSCRIPT_DIR.glob("*.docx"))

if not docx_files:
    raise FileNotFoundError(
        f"No DOCX transcript found inside: {TRANSCRIPT_DIR.resolve()}"
    )

TRANSCRIPT_PATH = docx_files[0]

print(f"Using transcript: {TRANSCRIPT_PATH}")


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# TRANSCRIPT_PATH = r"data/transcripts/transcript1.docx"



OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "parsed_transcript.json"
)


# ---------------------------------------------------------
# PARSING CONFIGURATION
# ---------------------------------------------------------

# Number of segments before and after the current segment
# that will be included in a conversational window.

CONTEXT_BEFORE = 2
CONTEXT_AFTER = 2


# ---------------------------------------------------------
# SPEAKER / TIMESTAMP PATTERN
# ---------------------------------------------------------

# Example:
#
# Amit Gogate   0:16
# Rajshekar Chavakula   13:59
#
# We support:
#   0:16
#   13:59
#   1:02:30
#
SPEAKER_PATTERN = re.compile(
    r"^\s*(.+?)\s+(\d{1,2}:\d{2}(?::\d{2})?)\s*$"
)


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def clean_text(text):
    """
    Perform minimal cleaning.

    Important:
    We deliberately DO NOT remove words such as:
    Yes, Yeah, No, Okay, Mhm, Right, etc.

    These may contain important conversational meaning.
    """

    if not text:
        return ""

    # Replace non-breaking spaces
    text = text.replace("\xa0", " ")

    # Normalize line breaks inside a speaker turn
    text = re.sub(r"\s*\n\s*", " ", text)

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------
# PARSE SPEAKER HEADER
# ---------------------------------------------------------

def parse_speaker_header(line):
    """
    Try to extract speaker name and timestamp.

    Returns:
        (speaker, timestamp)

    If the line is not a speaker header:
        (None, None)
    """

    match = SPEAKER_PATTERN.match(line)

    if not match:
        return None, None

    speaker = match.group(1).strip()
    timestamp = match.group(2).strip()

    return speaker, timestamp


# ---------------------------------------------------------
# EXTRACT TRANSCRIPT
# ---------------------------------------------------------

def extract_segments(transcript_path):
    """
    Read the DOCX and convert speaker paragraphs
    into structured transcript segments.
    """

    if not transcript_path.exists():
        raise FileNotFoundError(
            f"Transcript not found: {transcript_path}"
        )

    print(
        f"Loading transcript:\n{transcript_path}",
        flush=True
    )

    document = Document(transcript_path)

    print(
        f"Total DOCX paragraphs: {len(document.paragraphs)}",
        flush=True
    )

    segments = []

    segment_number = 1

    for paragraph in document.paragraphs:

        raw_text = paragraph.text.strip()

        if not raw_text:
            continue

        # -------------------------------------------------
        # Each paragraph may contain:
        #
        # Speaker + timestamp
        # followed by
        # spoken text
        # -------------------------------------------------

        lines = [
            line.strip()
            for line in raw_text.splitlines()
            if line.strip()
        ]

        if not lines:
            continue

        speaker = None
        timestamp = None
        text_lines = []

        # -------------------------------------------------
        # Check first line for speaker + timestamp
        # -------------------------------------------------

        first_speaker, first_timestamp = parse_speaker_header(
            lines[0]
        )

        if first_speaker:

            speaker = first_speaker
            timestamp = first_timestamp

            # Everything after speaker/timestamp
            # is spoken text.
            text_lines = lines[1:]

        else:
            # -------------------------------------------------
            # This is a continuation paragraph.
            #
            # In our current transcript structure this should
            # be uncommon, but we preserve it rather than
            # throwing information away.
            # -------------------------------------------------

            text_lines = lines

        text = clean_text(" ".join(text_lines))

        # Ignore document-level metadata
        if not speaker:
            continue

        # Ignore empty speaker turns
        if not text:
            continue

        segment = {
            "segment_id": f"SEG{segment_number:04d}",
            "speaker": speaker,
            "timestamp": timestamp,
            "text": text
        }

        segments.append(segment)

        segment_number += 1

    return segments


# ---------------------------------------------------------
# CREATE CONVERSATIONAL WINDOWS
# ---------------------------------------------------------

def create_conversational_windows(
    segments,
    context_before=CONTEXT_BEFORE,
    context_after=CONTEXT_AFTER
):
    """
    Create overlapping conversational windows.

    Example with:

        context_before = 2
        context_after = 2

    For SEG0005:

        SEG0003
        SEG0004
        SEG0005
        SEG0006
        SEG0007

    The original segments are preserved separately.
    """

    windows = []

    total_segments = len(segments)

    for index in range(total_segments):

        start_index = max(
            0,
            index - context_before
        )

        end_index = min(
            total_segments,
            index + context_after + 1
        )

        window_segments = segments[
            start_index:end_index
        ]

        window_text_parts = []

        for segment in window_segments:

            formatted_segment = (
                f"{segment['speaker']} "
                f"[{segment['timestamp']}]: "
                f"{segment['text']}"
            )

            window_text_parts.append(
                formatted_segment
            )

        window = {
            "window_id": f"WIN{index + 1:04d}",

            "center_segment_id":
                segments[index]["segment_id"],

            "segment_ids": [
                segment["segment_id"]
                for segment in window_segments
            ],

            "text": "\n".join(window_text_parts)
        }

        windows.append(window)

    return windows


# ---------------------------------------------------------
# EXTRACT MEETING METADATA
# ---------------------------------------------------------

def extract_metadata(document):
    """
    Extract basic meeting metadata from the first
    few paragraphs of the DOCX.
    """

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs[:10]
        if paragraph.text.strip()
    ]

    metadata = {
        "title": None,
        "date": None,
        "duration": None
    }

    if len(paragraphs) >= 1:
        metadata["title"] = paragraphs[0]

    if len(paragraphs) >= 2:
        metadata["date"] = paragraphs[1]

    if len(paragraphs) >= 3:
        metadata["duration"] = paragraphs[2]

    return metadata


# ---------------------------------------------------------
# SAVE PARSED TRANSCRIPT
# ---------------------------------------------------------

def save_parsed_transcript(
    metadata,
    segments,
    windows,
    output_path=OUTPUT_PATH
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    parsed_transcript = {

        "metadata": metadata,

        "parser": {
            "context_before": CONTEXT_BEFORE,
            "context_after": CONTEXT_AFTER,
            "total_segments": len(segments),
            "total_windows": len(windows)
        },

        "segments": segments,

        "windows": windows
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            parsed_transcript,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\nParsed transcript saved to:\n{output_path}",
        flush=True
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("SEPG TRANSCRIPT PARSER")
    print("=" * 60)

    print("\nTranscript:")
    print(TRANSCRIPT_PATH)

    # -----------------------------------------------------
    # Load document
    # -----------------------------------------------------

    if not TRANSCRIPT_PATH.exists():

        raise FileNotFoundError(
            f"\nTranscript file not found:\n"
            f"{TRANSCRIPT_PATH}\n\n"
            f"Make sure the DOCX is inside:\n"
            f"{PROJECT_ROOT / 'data' / 'transcripts'}"
        )

    document = Document(TRANSCRIPT_PATH)

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    metadata = extract_metadata(document)

    print("\nMeeting metadata:")
    print("-" * 60)

    print(f"Title:    {metadata['title']}")
    print(f"Date:     {metadata['date']}")
    print(f"Duration: {metadata['duration']}")

    # -----------------------------------------------------
    # Extract segments
    # -----------------------------------------------------

    print("\nExtracting conversation segments...")

    segments = extract_segments(
        TRANSCRIPT_PATH
    )

    print(
        f"\nTotal conversation segments: "
        f"{len(segments)}"
    )

    # -----------------------------------------------------
    # Create windows
    # -----------------------------------------------------

    print("\nCreating conversational windows...")

    windows = create_conversational_windows(
        segments
    )

    print(
        f"Total conversational windows: "
        f"{len(windows)}"
    )

    # -----------------------------------------------------
    # Display sample segments
    # -----------------------------------------------------

    print("\nFirst 5 segments:")
    print("-" * 60)

    for segment in segments[:5]:

        print(
            f"{segment['segment_id']} | "
            f"{segment['speaker']} | "
            f"{segment['timestamp']}"
        )

        print(
            f"  {segment['text']}"
        )

    # -----------------------------------------------------
    # Display sample window
    # -----------------------------------------------------

    if windows:

        print("\nSample conversational window:")
        print("-" * 60)

        print(windows[0]["text"])

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    save_parsed_transcript(
        metadata,
        segments,
        windows
    )

    print("\n" + "=" * 60)
    print("Transcript parsing completed successfully.")
    print("=" * 60)


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    main()