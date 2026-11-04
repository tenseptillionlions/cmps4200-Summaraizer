import json
import tempfile
import unittest
from pathlib import Path
import customtkinter as ctk
from tkinter import Label, Text, Tk
from unittest.mock import patch

from mock_ai import mock_summary, mock_transcription
from summaraizer_app import (
    COLORS,
    Compilation,
    CompilationStore,
    ScrollableFrame,
    SummaraizerApp,
    aggregate_summaries,
    classify_file,
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
        self.assertEqual(sample.count("Sample transcription for"), 1)
        self.assertEqual(sample.count("[0:30]"), 1)

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


class GuiIntuitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Tk()
        self.root.withdraw()
        self.store = CompilationStore(Path(self.temp_dir.name) / "compilations.json")
        self.app = SummaraizerApp(self.root, self.store)
        self.compilation = Compilation.create("TestComp")
        self.app.compilations.append(self.compilation)
        self.app._set_active(self.compilation)

    def tearDown(self) -> None:
        self.root.update_idletasks()
        self.root.destroy()
        self.temp_dir.cleanup()

    def _find_buttons(self, widget: object) -> list[ctk.CTkButton]:
        res: list[ctk.CTkButton] = []
        if isinstance(widget, ctk.CTkButton):
            res.append(widget)
        if hasattr(widget, "winfo_children"):
            for child in widget.winfo_children():
                res.extend(self._find_buttons(child))
        return res

    def _find_texts(self, widget: object) -> list[Text]:
        res: list[Text] = []
        if isinstance(widget, Text):
            res.append(widget)
        if hasattr(widget, "winfo_children"):
            for child in widget.winfo_children():
                res.extend(self._find_texts(child))
        return res

    def _find_labels(self, widget: object) -> list[Label]:
        res: list[Label] = []
        if isinstance(widget, Label):
            res.append(widget)
        if hasattr(widget, "winfo_children"):
            for child in widget.winfo_children():
                res.extend(self._find_labels(child))
        return res

    def test_drag_and_drop_on_viewer_page(self) -> None:
        self.app.show_page("viewer")
        doc_path = Path(self.temp_dir.name) / "doc1.txt"
        doc_path.write_text("Test document", encoding="utf-8")
        audio_path = Path(self.temp_dir.name) / "recording.wav"
        audio_path.write_text("audio data", encoding="utf-8")

        # Drop files on viewer page
        self.app._on_files_dropped([str(doc_path), str(audio_path)])
        self.assertIn(str(doc_path.resolve()), self.compilation.files)
        self.assertIn(str(audio_path.resolve()), self.compilation.files)
        # Dropping audio file automatically navigates to transcription
        self.assertEqual(self.app.active_page, "transcription")

    def test_open_file_capabilities(self) -> None:
        self.app.show_page("viewer")
        doc_path = Path(self.temp_dir.name) / "doc1.txt"
        doc_path.write_text("Sample file content", encoding="utf-8")
        self.app._add_file_paths([doc_path])
        self.app.show_page("viewer")

        # File list should have an Open button
        buttons = self._find_buttons(self.app.page_container)
        open_buttons = [b for b in buttons if b.cget("text") == "Open"]
        self.assertTrue(len(open_buttons) > 0)

        # Calling _open_file with system launcher
        with patch("os.startfile", create=True) as mock_start:
            self.app._open_file(str(doc_path))
            mock_start.assert_called_once_with(str(doc_path.resolve()))

    def test_mock_transcription_and_save_edits_button(self) -> None:
        audio_path = Path(self.temp_dir.name) / "audio.mp3"
        audio_path.write_text("audio bytes", encoding="utf-8")
        self.compilation.files.append(str(audio_path.resolve()))

        # Mock transcription generated upon opening without pressing generate
        self.app.show_page("transcription")
        resolved = str(audio_path.resolve())
        self.assertIn(resolved, self.app.transcription_drafts)
        self.assertTrue(len(self.app.transcription_drafts[resolved]) > 0)

        # Primary button is "Save Edits"
        buttons = self._find_buttons(self.app.page_container)
        save_edits_btns = [b for b in buttons if b.cget("text") == "Save Edits"]
        self.assertEqual(len(save_edits_btns), 1)
        save_edits_btn = save_edits_btns[0]
        self.assertEqual(save_edits_btn.cget("fg_color"), COLORS["success"])

        # User edits transcription and saves
        self.assertIsNotNone(self.app.transcription_editor)
        self.assertEqual(str(self.app.transcription_editor.cget("state")), "normal")
        self.app.transcription_editor.delete("1.0", "end")
        self.app.transcription_editor.insert("1.0", "User edited transcription")
        self.app._save_transcription(resolved)

        transcript_file = transcript_path_for(audio_path)
        self.assertTrue(transcript_file.exists())
        self.assertEqual(
            transcript_file.read_text(encoding="utf-8"),
            "User edited transcription",
        )
        self.assertEqual(self.app.active_page, "viewer")

    def test_split_screen_and_summary_buttons_flow(self) -> None:
        f1 = Path(self.temp_dir.name) / "file1.txt"
        f1.write_text("Content of file 1", encoding="utf-8")
        f2 = Path(self.temp_dir.name) / "file2.txt"
        f2.write_text("Content of file 2", encoding="utf-8")
        f3 = Path(self.temp_dir.name) / "audio.mp3"
        f3.write_text("audio data", encoding="utf-8")
        self.compilation.files = [
            str(f1.resolve()),
            str(f2.resolve()),
            str(f3.resolve()),
        ]

        # Generate mock summaries upon opening
        self.app.show_page("summary")
        self.assertIn(str(f1.resolve()), self.app.summary_drafts)
        self.assertIn(str(f2.resolve()), self.app.summary_drafts)

        # Split-screen comparison format: source text and summary editor
        texts = self._find_texts(self.app.page_container)
        self.assertGreaterEqual(len(texts), 2)
        text_contents = [t.get("1.0", "end-1c") for t in texts]
        self.assertTrue(any("Content of file 1" in c for c in text_contents))

        # Cancel button is present on summary generation screen
        buttons = self._find_buttons(self.app.page_container)
        btn_texts = [b.cget("text") for b in buttons]
        self.assertIn("Cancel", btn_texts)

        # Primary button is "Next File" when next file exists; Save button present
        self.assertEqual(self.app.summary_index, 0)
        self.assertIn("Next File", btn_texts)
        self.assertIn("Save", btn_texts)
        self.assertNotIn("Finalize", btn_texts)
        next_btn = next(b for b in buttons if b.cget("text") == "Next File")
        self.assertEqual(next_btn.cget("fg_color"), COLORS["accent"])
        self.assertEqual(int(next_btn.cget("height")), 44)
        for action_text in ("Open file", "Regenerate"):
            action_btn = next(b for b in buttons if b.cget("text") == action_text)
            self.assertEqual(action_btn.cget("fg_color"), COLORS["accent"])
            self.assertEqual(int(action_btn.cget("height")), 36)
        self.assertNotIn("Reset to mock", btn_texts)
        save_btn = next(b for b in buttons if b.cget("text") == "Save")
        cancel_btn = next(b for b in buttons if b.cget("text") == "Cancel")
        self.assertEqual(save_btn.master, cancel_btn.master)
        self.assertEqual(save_btn.master.pack_info()["side"], "left")
        self.assertEqual(next_btn.master.pack_info()["side"], "right")

        # Click Next File
        self.app._next_summary_file()
        self.assertEqual(self.app.summary_index, 1)

        # Last file: remove "save" button, turn next file into "Finalize" button
        buttons2 = self._find_buttons(self.app.page_container)
        btn_texts2 = [b.cget("text") for b in buttons2]
        self.assertIn("Finalize", btn_texts2)
        self.assertNotIn("Save", btn_texts2)
        self.assertNotIn("Next File", btn_texts2)
        self.assertIn("Cancel", btn_texts2)
        previous_btn = next(b for b in buttons2 if b.cget("text") == "Previous")
        finalize_btn = next(b for b in buttons2 if b.cget("text") == "Finalize")
        self.assertEqual(previous_btn.master, finalize_btn.master)
        self.assertEqual(previous_btn.master.pack_info()["side"], "right")
        self.assertEqual(finalize_btn.cget("fg_color"), COLORS["accent"])
        self.assertEqual(int(finalize_btn.cget("height")), 44)

        # Finalize compilation
        self.app._finalize_summary()
        self.assertNotEqual(self.compilation.master_summary, "")
        self.assertEqual(self.app.active_page, "viewer")

        # Once summary is finalized, buttons should still be visible:
        # Deep Dive is primary, Transcript viewing and Summary Generation still available
        viewer_buttons = self._find_buttons(self.app.page_container)
        viewer_map = {b.cget("text"): b for b in viewer_buttons}
        self.assertIn("Deep Dive", viewer_map)
        self.assertEqual(viewer_map["Deep Dive"].cget("fg_color"), COLORS["accent"])
        self.assertIn("Summary Generation", viewer_map)
        self.assertIn("Transcript viewing", viewer_map)

        # A finalized summary should not add duplicate footer actions or status.
        self.app.show_page("summary")
        finalized_buttons = self._find_buttons(self.app.page_container)
        finalized_texts = [b.cget("text") for b in finalized_buttons]
        self.assertEqual(finalized_texts.count("Cancel"), 1)
        self.assertNotIn("View master summary", finalized_texts)
        self.assertNotIn("Export master summary", finalized_texts)
        finalized_labels = [
            label.cget("text") for label in self._find_labels(self.app.page_container)
        ]
        self.assertNotIn("Master summary finalized", finalized_labels)

    def test_summary_regeneration_streams_into_editor(self) -> None:
        source = Path(self.temp_dir.name) / "streamed.txt"
        source.write_text("A source document.", encoding="utf-8")
        source_path = str(source.resolve())
        self.compilation.files = [source_path]

        self.app.show_page("summary")
        self.app._generate_summary(source_path)

        generated = self.app.summary_drafts[source_path]
        self.assertIsNotNone(self.app.summary_progress)
        self.assertLess(
            len(self.app.summary_editor.get("1.0", "end-1c")),
            len(generated),
        )
        while self.app.summary_streaming_path is not None:
            if self.app.summary_stream_after is not None:
                self.root.after_cancel(self.app.summary_stream_after)
                self.app.summary_stream_after = None
            offset = len(self.app.summary_editor.get("1.0", "end-1c"))
            self.app._stream_summary_text(
                source_path,
                generated,
                self.app.summary_stream_generation,
                offset,
            )

        self.assertIsNone(self.app.summary_streaming_path)
        self.assertEqual(
            self.app.summary_editor.get("1.0", "end-1c"),
            generated,
        )

    def test_chat_page_keeps_send_flow(self) -> None:
        self.app.show_page("chat")
        self.app.chat_entry.insert("1.0", "What is in this compilation?")

        self.app._send_chat_message()

        self.assertEqual(self.app.active_page, "chat")
        self.assertEqual(self.app.chat_messages[0][0], "You")
        self.assertEqual(self.app.chat_messages[0][1], "What is in this compilation?")

    def test_sidebar_keeps_workflow_navigation_and_compilations_visible(self) -> None:
        self.app.show_page("viewer")
        nav_text = [
            widget.cget("text") for widget in self.app.sidebar_nav.winfo_children()
        ]
        compilation_text = [
            widget.cget("text")
            for widget in self.app.sidebar_compilations.content.winfo_children()
        ]

        self.assertEqual(
            nav_text,
            ["Home", "Files", "Transcription", "Summary", "Deep Dive"],
        )
        self.assertTrue(any("TestComp" in text for text in compilation_text))
        self.app.show_page("viewer")
        self.assertEqual(len(self.app.sidebar_compilations.content.winfo_children()), 1)

    def test_shell_removes_duplicate_header_and_uses_large_empty_state_action(self) -> None:
        self.app.compilations.clear()
        self.app.active_compilation = None
        self.app.show_page("home")

        self.assertFalse(hasattr(self.app, "page_title"))
        buttons = self._find_buttons(self.app.page_container)
        create_button = next(
            button
            for button in buttons
            if button.cget("text") == "+  Create your first compilation"
        )
        self.assertGreaterEqual(int(create_button.cget("height")), 56)
        self.assertGreaterEqual(int(create_button.cget("width")), 300)

    def test_scrollbar_hides_when_not_needed_and_appears_for_overflow(self) -> None:
        scrollable = ScrollableFrame(self.root)
        scrollable.pack(fill="both", expand=True)
        self.root.update_idletasks()
        self.assertFalse(scrollable.scrollbar.winfo_manager())

        for index in range(40):
            Label(scrollable.content, text=f"Row {index}").pack()
        self.root.update_idletasks()
        self.assertEqual(scrollable.scrollbar.winfo_manager(), "pack")
        scrollable.destroy()
        self.root.update_idletasks()

    def test_mousewheel_scrolls_when_pointer_is_over_a_list_item(self) -> None:
        scrollable = ScrollableFrame(self.root)
        scrollable.pack(fill="both", expand=True)
        item = None
        for index in range(40):
            item = Label(scrollable.content, text=f"Row {index}")
            item.pack()
        self.root.update_idletasks()

        self.assertIsNotNone(item)
        self.assertLessEqual(float(scrollable.canvas.yview()[0]), 0.001)
        item.event_generate("<MouseWheel>", delta=-120)
        self.root.update_idletasks()
        self.assertGreater(float(scrollable.canvas.yview()[0]), 0)

    def test_sidebar_compilation_list_scrolls_when_it_overflows(self) -> None:
        self.app.compilations = [
            Compilation.create(f"Collection {index}") for index in range(30)
        ]
        self.app._refresh_sidebar()
        self.root.update_idletasks()

        self.assertEqual(self.app.sidebar_compilations.scrollbar.winfo_manager(), "pack")

    def test_chat_retry_remains_available_after_multiple_retries(self) -> None:
        self.app.show_page("chat")
        self.app.chat_entry.insert("1.0", "What is in this compilation?")
        self.app._send_chat_message()
        response_index = len(self.app.chat_messages) - 1

        self.app._retry_chat_response(response_index)
        self.assertIn(response_index, self.app.retryable_chat_responses)
        self.app._retry_chat_response(response_index)
        self.assertIn(response_index, self.app.retryable_chat_responses)
        retry_buttons = [
            button
            for button in self._find_buttons(self.app.page_container)
            if button.cget("text") == "Retry"
        ]
        self.assertEqual(len(retry_buttons), 1)

if __name__ == "__main__":
    unittest.main()
