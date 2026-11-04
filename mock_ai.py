from __future__ import annotations

from pathlib import Path


def mock_transcription(path: str | Path) -> tuple[str, bool]:
    name = Path(path).stem
    lowered_name = name.lower()
    if any(term in lowered_name for term in ("silent", "empty", "no-speech")):
        return (
            "No speech could be detected in this recording. Check the audio file "
            "and try again.",
            True,
        )
    if any(term in lowered_name for term in ("unclear", "noise", "low-confidence")):
        return (
            f"[Low confidence] A short sample transcript was generated for "
            f"“{name}”. Background noise or unclear speech may affect accuracy.",
            True,
        )
    transcript = (
        f"[0:00]Sample transcription for “{name}”.\n"
        "[0:30] This also demonstrates the timestamps.\n"
        "[1:00] I am taking very long pauses. That is why the timestamps are like this.\n"
        "[1:30]Good result."
    )
    return transcript, False


def mock_summary(path: str | Path, transcription: str = "") -> str:
    name = Path(path).stem
    if transcription.startswith("No speech could be detected"):
        detail = "No speech was detected, so this file has no transcript to summarize."
    elif transcription.startswith("[Low confidence]"):
        detail = (
            "A sample transcript was available, but its low confidence means "
            "details should be verified against the recording."
        )
    elif name.casefold() == "classnotes":
        detail = "A summary of notes for the class."
    elif name.casefold().endswith("-transcript"):
        detail = "Summary of the audio transcription. \n"
    else:
        detail = f"Sample summary for {name}."
    return f"{name}: {detail} (Sample summary — AI is not connected.)"
