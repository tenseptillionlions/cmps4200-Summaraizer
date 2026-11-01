import json
import tempfile
import unittest
from pathlib import Path

from summaraizer_app import (
    Compilation,
    CompilationStore,
    aggregate_summaries,
    classify_file,
    mock_summary,
    mock_transcription,
    transcript_path_for,
)


class FileTypeTests(unittest.TestCase):
    def test_classifies_supported_document_and_audio_extensions(self) -> None:
        self.assertEqual(classify_file("notes.PDF"), "PDF")
        self.assertEqual(classify_file("recording.MP3"), "Audio")
        self.assertEqual(classify_file("recording.wav"), "Audio")
        self.assertIsNone(classify_file("notes.docx"))
        self.assertIsNone(classify_file("archive.zip"))

    def test_transcription_has_normal_and_low_confidence_samples(self) -> None:
        sample, low_confidence = mock_transcription("interview.mp3")
        self.assertIn("interview", sample)
        self.assertFalse(low_confidence)
        self.assertGreater(sample.count("\n\n"), 5)

        sample, low_confidence = mock_transcription("unclear-audio.wav")
        self.assertIn("Low confidence", sample)
        self.assertTrue(low_confidence)

        sample, low_confidence = mock_transcription("silent-room.wav")
        self.assertIn("No speech could be detected", sample)
        self.assertTrue(low_confidence)

    def test_summary_uses_transcription_state(self) -> None:
        no_speech, _ = mock_transcription("empty.wav")
        self.assertIn("no transcript", mock_summary("empty.wav", no_speech))
        self.assertIn("AI is not connected", mock_summary("notes.pdf"))
        self.assertIn(
            "audio transcription",
            mock_summary("recording-transcript.txt").lower(),
        )

    def test_transcript_path_replaces_audio_extension(self) -> None:
        self.assertEqual(
            transcript_path_for(Path("recording.mp3")).name,
            "recording-transcript.txt",
        )

    def test_only_text_and_pdf_files_are_summarized(self) -> None:
        compilation = Compilation.create("Research")
        compilation.files = [
            "recording.mp3",
            "notes.txt",
            "recording-transcript.txt",
            "scan.pdf",
            "recording.wav",
        ]
        self.assertEqual(
            [Path(path).name for path in compilation.summary_files],
            ["notes.txt", "recording-transcript.txt", "scan.pdf"],
        )


class CompilationStoreTests(unittest.TestCase):
    def test_round_trips_compilation_and_aggregates_summaries_in_file_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CompilationStore(Path(directory) / "compilations.json")
            compilation = Compilation.create("Research")
            compilation.files = ["Zulu.pdf", "Alpha-transcript.txt"]
            compilation.summaries = {
                "Zulu.pdf": "Second summary",
                "Alpha-transcript.txt": "First summary",
            }
            compilation.accepted_transcriptions = ["Alpha.mp3"]
            compilation.master_summary = aggregate_summaries(compilation)
            store.save([compilation])

            restored = store.load()[0]
            self.assertEqual(restored.name, "Research")
            self.assertEqual(
                restored.files, ["Zulu.pdf", "Alpha-transcript.txt"]
            )
            self.assertEqual(restored.accepted_transcriptions, ["Alpha.mp3"])
            self.assertEqual(
                restored.master_summary,
                "Compilation summary: Research\n\nFirst summary\n\nSecond summary",
            )
            summary_path = store.write_summary(restored)
            self.assertEqual(summary_path.name, "Summary.txt")
            self.assertEqual(
                summary_path.read_text(encoding="utf-8"),
                restored.master_summary,
            )

    def test_rejects_malformed_store_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compilations.json"
            path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                CompilationStore(path).load()


if __name__ == "__main__":
    unittest.main()
