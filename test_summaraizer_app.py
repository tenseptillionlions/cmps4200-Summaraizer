import json
import queue
import sys
import tempfile
import types
import unittest
from pathlib import Path
import customtkinter as ctk
from tkinter import Label, Text, Tk
from unittest.mock import patch

import ai_backend
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


class LocalAIBackendTests(unittest.TestCase):
    def test_uses_models_from_the_sample_implementations(self) -> None:
        self.assertEqual(ai_backend.TRANSCRIPTION_MODEL, "openai/whisper-medium")
        self.assertEqual(ai_backend.TEXT_MODEL, "Qwen/Qwen3-4B-Instruct-2507")
        self.assertEqual(ai_backend.CPU_TRANSCRIPTION_MODEL, "openai/whisper-small")
        self.assertEqual(ai_backend.CPU_TEXT_MODEL, "Qwen/Qwen3-0.6B")

    def test_cpu_generator_uses_small_model_without_disk_offloading(self) -> None:
        captured: dict[str, object] = {}
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            float16="float16",
            float32="float32",
        )

        def pipeline(task: str, **kwargs: object) -> str:
            captured.update(task=task, **kwargs)
            return "cpu-generator"

        with patch.object(ai_backend, "_generator", None), patch.object(
            ai_backend, "_load_runtime", return_value=(torch, pipeline)
        ):
            self.assertEqual(ai_backend._get_generator(), "cpu-generator")

        self.assertEqual(captured["model"], ai_backend.CPU_TEXT_MODEL)
        self.assertEqual(captured["device"], -1)
        self.assertNotIn("device_map", captured)

    def test_limited_cuda_memory_uses_small_text_model_on_gpu(self) -> None:
        captured: dict[str, object] = {}
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                mem_get_info=lambda: (4 * 1024**3, 8 * 1024**3),
            ),
            float16="float16",
            float32="float32",
        )

        def pipeline(task: str, **kwargs: object) -> str:
            captured.update(task=task, **kwargs)
            return "small-gpu-generator"

        with patch.object(ai_backend, "_generator", None), patch.object(
            ai_backend, "_load_runtime", return_value=(torch, pipeline)
        ):
            self.assertEqual(ai_backend._get_generator(), "small-gpu-generator")

        self.assertEqual(captured["model"], ai_backend.CPU_TEXT_MODEL)
        self.assertEqual(captured["device"], 0)
        self.assertNotIn("device_map", captured)

    def test_sufficient_cuda_memory_keeps_sample_text_model(self) -> None:
        captured: dict[str, object] = {}
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                mem_get_info=lambda: (12 * 1024**3, 16 * 1024**3),
            ),
            float16="float16",
            float32="float32",
        )

        def pipeline(task: str, **kwargs: object) -> str:
            captured.update(task=task, **kwargs)
            return "large-gpu-generator"

        with patch.object(ai_backend, "_generator", None), patch.object(
            ai_backend, "_load_runtime", return_value=(torch, pipeline)
        ):
            self.assertEqual(ai_backend._get_generator(), "large-gpu-generator")

        self.assertEqual(captured["model"], ai_backend.TEXT_MODEL)
        self.assertEqual(captured["device_map"], "auto")
        self.assertNotIn("device", captured)

    def test_cpu_transcriber_selects_small_whisper_model(self) -> None:
        captured: dict[str, object] = {}
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            float16="float16",
            float32="float32",
        )

        def pipeline(task: str, **kwargs: object) -> str:
            captured.update(task=task, **kwargs)
            return "cpu-transcriber"

        with patch.object(ai_backend, "_transcriber", None), patch.object(
            ai_backend, "_load_runtime", return_value=(torch, pipeline)
        ):
            self.assertEqual(ai_backend._get_transcriber(), "cpu-transcriber")

        self.assertEqual(captured["model"], ai_backend.CPU_TRANSCRIPTION_MODEL)
        self.assertEqual(captured["device"], -1)

    def test_limited_cuda_memory_uses_small_whisper_model_on_gpu(self) -> None:
        captured: dict[str, object] = {}
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                mem_get_info=lambda: (2 * 1024**3, 8 * 1024**3),
            ),
            float16="float16",
            float32="float32",
        )

        def pipeline(task: str, **kwargs: object) -> str:
            captured.update(task=task, **kwargs)
            return "small-gpu-transcriber"

        with patch.object(ai_backend, "_transcriber", None), patch.object(
            ai_backend, "_load_runtime", return_value=(torch, pipeline)
        ):
            self.assertEqual(
                ai_backend._get_transcriber(), "small-gpu-transcriber"
            )

        self.assertEqual(captured["model"], ai_backend.CPU_TRANSCRIPTION_MODEL)
        self.assertEqual(captured["device"], 0)

    def test_reads_text_documents_and_rejects_blank_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notes.txt"
            path.write_text("Important local notes.", encoding="utf-8")
            self.assertEqual(ai_backend.read_document_text(path), "Important local notes.")
            path.write_text("  \n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No readable text"):
                ai_backend.read_document_text(path)

    def test_selects_embedding_device_instead_of_disk_offload_device(self) -> None:
        cpu_device = types.SimpleNamespace(type="cpu")
        embeddings = types.SimpleNamespace(weight=types.SimpleNamespace(device=cpu_device))

        class OffloadedModel:
            device = "disk"

            def get_input_embeddings(self) -> object:
                return embeddings

            def named_modules(self) -> list[tuple[str, object]]:
                return [("model.embed_tokens", embeddings)]

        selected = ai_backend._input_device(
            OffloadedModel(),
            types.SimpleNamespace(device=lambda device: device),
        )
        self.assertIs(selected, cpu_device)

    def test_normalizes_string_disk_embedding_to_a_compute_device(self) -> None:
        embeddings = types.SimpleNamespace(weight=types.SimpleNamespace(device="disk"))

        class OffloadedModel:
            device = "disk"
            hf_device_map = {
                "model.embed_tokens": "disk",
                "model.layers.0": "disk",
            }

            def get_input_embeddings(self) -> object:
                return embeddings

            def named_modules(self) -> list[tuple[str, object]]:
                return [("model.embed_tokens", embeddings)]

        def make_device(candidate: object) -> object:
            return types.SimpleNamespace(type=str(candidate).split(":")[0])

        selected = ai_backend._input_device(
            OffloadedModel(),
            types.SimpleNamespace(device=make_device),
        )
        self.assertEqual(selected.type, "cpu")

    def test_pipeline_runtime_patch_handles_disk_offloaded_device_map(self) -> None:
        ai_backend._patch_runtime_for_disk_offload()
        from accelerate.utils import offload as accelerate_offload

        # Verify OffloadedWeightsLoader tied weight fallback
        loader = accelerate_offload.OffloadedWeightsLoader(
            state_dict={"model.embed_tokens.weight": "tensor_model_embed"}
        )
        self.assertEqual(loader["lm_head.weight"], "tensor_model_embed")

    def test_summary_and_deep_dive_use_separate_source_grounded_prompts(self) -> None:
        with patch.object(ai_backend, "_generate", return_value="Generated locally") as generate:
            self.assertEqual(
                ai_backend.summarize_text("Source details.", "notes.txt"),
                "Generated locally",
            )
            summary_messages = generate.call_args.args[0]
            self.assertIn("notes.txt", summary_messages[1]["content"])
            self.assertIn("Source details.", summary_messages[1]["content"])

            self.assertEqual(
                ai_backend.answer_question("What happened?", "A local event."),
                "Generated locally",
            )
            question_messages = generate.call_args.args[0]
            self.assertIn("A local event.", question_messages[1]["content"])
            self.assertIn("What happened?", question_messages[1]["content"])
            self.assertFalse(generate.call_args.kwargs["enable_thinking"])

    def test_summary_prompt_limits_input_and_requests_one_short_paragraph(self) -> None:
        source = "A" * (ai_backend.SUMMARY_INPUT_MAX_CHARS + 500)
        with patch.object(ai_backend, "_generate", return_value="Short summary.") as generate:
            result = ai_backend.summarize_text(source, "long.txt")

        self.assertEqual(result, "Short summary.")
        prompt = generate.call_args.args[0]
        self.assertIn("exactly one concise paragraph", prompt[0]["content"])
        self.assertIn("4-5 short sentences", prompt[0]["content"])
        self.assertIn("quick readability", prompt[0]["content"])
        self.assertIn("40-70 words total", prompt[0]["content"])
        self.assertIn(
            source[: ai_backend.SUMMARY_INPUT_MAX_CHARS],
            prompt[1]["content"],
        )
        self.assertIn("Remaining source text omitted", prompt[1]["content"])
        self.assertEqual(
            generate.call_args.kwargs["max_new_tokens"],
            ai_backend.SUMMARY_MAX_NEW_TOKENS,
        )
        self.assertFalse(generate.call_args.kwargs["enable_thinking"])

    def test_transcription_formats_local_model_segments_with_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            path.write_bytes(b"audio")
            streamed: list[str] = []
            progress: list[str] = []
            librosa = types.ModuleType("librosa")
            librosa.load = lambda *_args, **_kwargs: ([0.0] * 4, 2)

            def recognize(_audio: object) -> dict[str, str]:
                return {"text": "Hello world"}

            with patch.object(
                ai_backend,
                "_get_transcriber",
                return_value=recognize,
            ), patch.dict(sys.modules, {"librosa": librosa}):
                self.assertEqual(
                    ai_backend.transcribe_audio(
                        path,
                        on_progress=progress.append,
                        on_chunk=streamed.append,
                    ),
                    "[0:00] Hello world",
                )
            self.assertEqual(streamed, ["[0:00] Hello world\n"])
            self.assertIn("Transcribing audio segment 1 of 1…", progress)

    def test_text_generation_streams_model_output_and_loading_status(self) -> None:
        streamed: list[str] = []
        progress: list[str] = []

        class FakeStreamer:
            def __init__(self, _tokenizer: object, **_kwargs: object) -> None:
                self.chunks: queue.Queue[str | None] = queue.Queue()

            def __iter__(self) -> object:
                return self

            def __next__(self) -> str:
                chunk = self.chunks.get(timeout=2)
                if chunk is None:
                    raise StopIteration
                return chunk

        class FakeInputs(dict):
            def to(self, _device: str) -> object:
                return self

        class FakeTokenizer:
            eos_token_id = 0

            def apply_chat_template(
                self, _messages: object, **_kwargs: object
            ) -> str:
                return "prompt"

            def __call__(self, _prompt: str, **_kwargs: object) -> FakeInputs:
                return FakeInputs()

        class FakeModel:
            device = "cpu"

            def generate(self, *, streamer: FakeStreamer, **_kwargs: object) -> None:
                for chunk in ("Hello", " ", "world"):
                    streamer.chunks.put(chunk)
                streamer.chunks.put(None)

        class FakeGenerator:
            model = FakeModel()
            tokenizer = FakeTokenizer()

        transformers = types.ModuleType("transformers")
        transformers.TextIteratorStreamer = FakeStreamer
        with patch.object(
            ai_backend, "_get_generator", return_value=FakeGenerator()
        ), patch.dict(sys.modules, {"transformers": transformers}):
            answer = ai_backend._generate(
                [{"role": "user", "content": "Question"}],
                max_new_tokens=8,
                on_progress=progress.append,
                on_chunk=streamed.append,
            )

        self.assertEqual(answer, "Hello world")
        self.assertEqual(streamed, ["Hello", " ", "world"])
        self.assertIn("Generating response…", progress)

    def test_model_loading_reports_runtime_download_and_ready_states(self) -> None:
        progress: list[str] = []
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch.float16 = "float16"
        torch.float32 = "float32"
        transformers = types.ModuleType("transformers")
        transformers.pipeline = lambda *_args, **_kwargs: object()
        with patch.object(ai_backend, "_generator", None), patch.dict(
            sys.modules, {"torch": torch, "transformers": transformers}
        ):
            ai_backend._get_generator(progress.append)

        self.assertIn("Preparing the local AI runtime…", progress)
        self.assertIn(
            "Loading Qwen 3 locally. First use may download model files.", progress
        )
        self.assertIn(
            "Loading the smaller Qwen/Qwen3-0.6B CPU model…", progress
        )
        self.assertIn("Qwen 3 is ready.", progress)


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

        def fake_transcribe(path, on_progress=None, on_chunk=None):
            transcript = f"Transcript for {Path(path).name}"
            if on_chunk:
                on_chunk(transcript + "\n")
            return transcript

        def fake_summarize(path, on_progress=None, on_chunk=None):
            summary = f"Local summary for {Path(path).name}. " * 4
            if on_chunk:
                on_chunk(summary)
            return summary

        def fake_answer(_question, _context, on_progress=None, on_chunk=None):
            answer = "Answer from local compilation context."
            if on_chunk:
                on_chunk("Answer from local ")
                on_chunk("compilation context.")
            return answer

        self.ai_patches = [
            patch("summaraizer_app.transcribe_audio", side_effect=fake_transcribe),
            patch("summaraizer_app.summarize_document", side_effect=fake_summarize),
            patch("summaraizer_app.answer_question", side_effect=fake_answer),
        ]
        for ai_patch in self.ai_patches:
            ai_patch.start()

        def run_ai_task(task, on_success, _error_title, on_error=None, on_progress=None, on_chunk=None):
            try:
                on_success(
                    task(
                        on_progress or (lambda _message: None),
                        on_chunk or (lambda _chunk: None),
                    )
                )
            except Exception as error:
                if on_error is not None:
                    on_error(error)
                else:
                    raise

        self.app._run_ai_task = run_ai_task
        self.compilation = Compilation.create("TestComp")
        self.app.compilations.append(self.compilation)
        self.app._set_active(self.compilation)

    def tearDown(self) -> None:
        self.root.update_idletasks()
        self.root.destroy()
        self.temp_dir.cleanup()
        for ai_patch in self.ai_patches:
            ai_patch.stop()

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

    def test_local_transcription_and_save_edits_button(self) -> None:
        audio_path = Path(self.temp_dir.name) / "audio.mp3"
        audio_path.write_text("audio bytes", encoding="utf-8")
        self.compilation.files.append(str(audio_path.resolve()))

        self.app.show_page("transcription")
        resolved = str(audio_path.resolve())
        self.assertNotIn(resolved, self.app.transcription_drafts)
        self.app._generate_transcription(resolved)
        self.assertIn(resolved, self.app.transcription_drafts)
        self.assertEqual(
            self.app.transcription_drafts[resolved],
            "Transcript for audio.mp3",
        )
        self.assertEqual(
            self.app.transcription_editor.get("1.0", "end-1c"),
            "Transcript for audio.mp3",
        )

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

        # Generate local summaries upon opening and moving between files.
        self.app.show_page("summary")
        self.assertIn(str(f1.resolve()), self.app.summary_drafts)

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
        self.assertNotIn("Reset sample", btn_texts)
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

    def test_summary_generation_streams_directly_into_editor(self) -> None:
        source = Path(self.temp_dir.name) / "streamed.txt"
        source.write_text("A source document.", encoding="utf-8")
        source_path = str(source.resolve())
        self.compilation.files = [source_path]

        self.app.show_page("summary")
        self.app._generate_summary(source_path)

        generated = self.app.summary_drafts[source_path]
        self.assertIsNotNone(self.app.summary_progress)
        self.assertIsNone(self.app.summary_streaming_path)
        self.assertEqual(
            self.app.summary_editor.get("1.0", "end-1c"),
            generated,
        )

    def test_chat_streams_answer_into_conversation(self) -> None:
        source = Path(self.temp_dir.name) / "chat.txt"
        source.write_text("The compilation discusses local testing.", encoding="utf-8")
        self.compilation.files = [str(source.resolve())]
        self.app.show_page("chat")
        self.app.chat_entry.insert("1.0", "What is in this compilation?")

        self.app._send_chat_message()

        self.assertEqual(
            self.app.chat_messages[-1],
            ("Summaraizer", "Answer from local compilation context."),
        )
        self.assertEqual(
            self.app.chat_message_labels[len(self.app.chat_messages) - 1].cget("text"),
            "Answer from local compilation context.",
        )

    def test_chat_page_keeps_send_flow(self) -> None:
        source = Path(self.temp_dir.name) / "chat.txt"
        source.write_text("The compilation discusses local testing.", encoding="utf-8")
        self.compilation.files = [str(source.resolve())]
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
        source = Path(self.temp_dir.name) / "chat.txt"
        source.write_text("The compilation discusses local testing.", encoding="utf-8")
        self.compilation.files = [str(source.resolve())]
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
