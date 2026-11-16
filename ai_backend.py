from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from queue import Empty
from threading import Lock, Thread
from typing import Any


TRANSCRIPTION_MODEL = "openai/whisper-medium"
CPU_TRANSCRIPTION_MODEL = "openai/whisper-small"
TEXT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
CPU_TEXT_MODEL = "Qwen/Qwen3-0.6B"
TRANSCRIPTION_CHUNK_SECONDS = 30
SUMMARY_INPUT_MAX_CHARS = 12000
SUMMARY_MAX_NEW_TOKENS = 200
GPU_TEXT_MODEL_MIN_FREE_GIB = 10
GPU_SMALL_TEXT_MODEL_MIN_FREE_GIB = 2
GPU_TRANSCRIPTION_MODEL_MIN_FREE_GIB = 3
GPU_SMALL_TRANSCRIPTION_MODEL_MIN_FREE_GIB = 1

ProgressCallback = Callable[[str], None]
TextCallback = Callable[[str], None]

_transcriber: Any = None
_generator: Any = None
_transcriber_lock = Lock()
_generator_lock = Lock()


def _patch_runtime_for_disk_offload() -> None:
    try:
        from transformers.pipelines import base as pipeline_base

        orig_pipeline_init = getattr(pipeline_base.Pipeline, "_unpatched_init", None)
        if orig_pipeline_init is None:
            orig_pipeline_init = pipeline_base.Pipeline.__init__
            pipeline_base.Pipeline._unpatched_init = orig_pipeline_init

            def safe_pipeline_init(self: Any, *args: Any, **kwargs: Any) -> None:
                model = kwargs.get("model", args[0] if args else getattr(self, "model", None))
                hf_map = getattr(model, "hf_device_map", None)
                restored: dict[str, Any] | None = None
                if isinstance(hf_map, dict):
                    restored = dict(hf_map)
                    for key, val in hf_map.items():
                        if str(val).casefold() == "disk":
                            hf_map[key] = "cpu"
                try:
                    return orig_pipeline_init(self, *args, **kwargs)
                finally:
                    if restored is not None and isinstance(hf_map, dict):
                        hf_map.clear()
                        hf_map.update(restored)

            pipeline_base.Pipeline.__init__ = safe_pipeline_init
    except (ImportError, AttributeError):
        pass

    try:
        from accelerate.utils import offload as accelerate_offload

        orig_loader_getitem = getattr(
            accelerate_offload.OffloadedWeightsLoader, "_unpatched_getitem", None
        )
        if orig_loader_getitem is None:
            orig_loader_getitem = accelerate_offload.OffloadedWeightsLoader.__getitem__
            accelerate_offload.OffloadedWeightsLoader._unpatched_getitem = (
                orig_loader_getitem
            )

            def safe_loader_getitem(self: Any, key: str) -> Any:
                if key not in self.state_dict and key not in self.index:
                    if key == "lm_head.weight":
                        for candidate in ("model.embed_tokens.weight", "embed_tokens.weight"):
                            if candidate in self.state_dict or candidate in self.index:
                                key = candidate
                                break
                return orig_loader_getitem(self, key)

            accelerate_offload.OffloadedWeightsLoader.__getitem__ = safe_loader_getitem
    except (ImportError, AttributeError):
        pass


def _load_runtime(on_progress: ProgressCallback | None = None) -> tuple[Any, Any]:
    if on_progress is not None:
        on_progress("Preparing the local AI runtime…")
    try:
        import torch
        from transformers import pipeline
    except ImportError as error:
        raise RuntimeError(
            "Local AI dependencies are missing. Install the packages in "
            "requirements.txt and try again."
        ) from error
    _patch_runtime_for_disk_offload()
    return torch, pipeline


def _cuda_free_memory_gib(torch: Any) -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
    except (AttributeError, RuntimeError):
        return 0.0
    return free_bytes / (1024**3)


def _get_transcriber(on_progress: ProgressCallback | None = None) -> Any:
    global _transcriber
    with _transcriber_lock:
        if _transcriber is None:
            torch, pipeline = _load_runtime(on_progress)
            if on_progress is not None:
                on_progress(
                    "Loading Whisper medium locally. First use may download model files."
                )
            free_gpu_gib = _cuda_free_memory_gib(torch)
            use_cuda = free_gpu_gib >= GPU_SMALL_TRANSCRIPTION_MODEL_MIN_FREE_GIB
            use_medium_model = free_gpu_gib >= GPU_TRANSCRIPTION_MODEL_MIN_FREE_GIB
            model_name = (
                TRANSCRIPTION_MODEL
                if use_medium_model
                else CPU_TRANSCRIPTION_MODEL
            )
            if on_progress is not None:
                on_progress(
                    f"Loading {model_name} on GPU…"
                    if use_cuda
                    else f"Loading the smaller {model_name} CPU model…"
                )
            _transcriber = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                torch_dtype=torch.float16 if use_cuda else torch.float32,
                device=0 if use_cuda else -1,
                chunk_length_s=30,
            )
            if on_progress is not None:
                on_progress("Whisper is ready.")
    return _transcriber


def _get_generator(on_progress: ProgressCallback | None = None) -> Any:
    global _generator
    with _generator_lock:
        if _generator is None:
            torch, pipeline = _load_runtime(on_progress)
            if on_progress is not None:
                on_progress(
                    "Loading Qwen 3 locally. First use may download model files."
                )
            free_gpu_gib = _cuda_free_memory_gib(torch)
            use_large_model = free_gpu_gib >= GPU_TEXT_MODEL_MIN_FREE_GIB
            use_small_gpu_model = (
                not use_large_model
                and free_gpu_gib >= GPU_SMALL_TEXT_MODEL_MIN_FREE_GIB
            )
            use_cuda = use_large_model or use_small_gpu_model
            model_name = (
                TEXT_MODEL
                if use_large_model
                else CPU_TEXT_MODEL
            )
            if on_progress is not None:
                on_progress(
                    f"Loading {model_name} on GPU…"
                    if use_cuda
                    else f"Loading the smaller {model_name} CPU model…"
                )
            pipeline_options: dict[str, Any] = {
                "model": model_name,
                "torch_dtype": torch.float16 if use_cuda else torch.float32,
            }
            if use_large_model:
                pipeline_options["device_map"] = "auto"
            elif use_small_gpu_model:
                pipeline_options["device"] = 0
            else:
                # Keep the smaller CPU model entirely in RAM. device_map="auto"
                # can put every weight on disk on memory-constrained systems.
                pipeline_options["device"] = -1
            _generator = pipeline("text-generation", **pipeline_options)
            if on_progress is not None:
                on_progress("Qwen 3 is ready.")
    return _generator


def read_document_text(path: str | Path) -> str:
    """Read supported local text or PDF files for summarization and Deep Dive."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Document not found: {target}")
    if target.suffix.lower() == ".txt":
        text = target.read_text(encoding="utf-8", errors="replace")
    elif target.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError(
                "PDF support is missing. Install the packages in requirements.txt "
                "and try again."
            ) from error
        from pypdf.errors import PdfReadError

        try:
            reader = PdfReader(str(target))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except (PdfReadError, OSError, EOFError) as error:
            raise ValueError(f"Unable to read PDF {target.name}: {error}") from error
    else:
        raise ValueError(f"Unsupported document type: {target.suffix or '(none)'}")
    if not text.strip():
        raise ValueError(f"No readable text was found in {target.name}.")
    return text.strip()


def _compute_device(candidate: Any, torch: Any) -> Any | None:
    if candidate is None:
        return None
    candidate_str = str(candidate).casefold().strip()
    if candidate_str in ("disk", "meta") or candidate_str.startswith(("disk:", "meta:")):
        return None
    if isinstance(candidate, int) and not isinstance(candidate, bool):
        candidate = f"cuda:{candidate}"
    try:
        device = torch.device(candidate)
    except (TypeError, RuntimeError, ValueError, Exception):
        return None
    if device.type in ("meta", "disk"):
        return None
    return device


def _input_device(model: Any, torch: Any) -> Any:
    """Choose a normalized compute device, even when model weights are offloaded."""
    get_embeddings = getattr(model, "get_input_embeddings", None)
    embeddings = get_embeddings() if callable(get_embeddings) else None
    embedding_device = getattr(getattr(embeddings, "weight", None), "device", None)
    device = _compute_device(embedding_device, torch)
    if device is not None:
        return device

    device_map = getattr(model, "hf_device_map", {})
    named_modules = getattr(model, "named_modules", lambda: [])()
    embedding_name = next(
        (name for name, module in named_modules if module is embeddings), ""
    )
    mapped_devices = [
        (name, device)
        for name, device in device_map.items()
        if embedding_name == name or embedding_name.startswith(f"{name}.")
    ]
    mapped_devices.sort(key=lambda item: len(item[0]), reverse=True)
    candidates = [device for _, device in mapped_devices]
    candidates.extend(
        device for device in device_map.values() if device != "disk"
    )
    candidates.append(getattr(model, "device", None))

    for candidate in candidates:
        device = _compute_device(candidate, torch)
        if device is not None:
            return device
    if "disk" in device_map.values():
        # Accelerate uses CPU as the execution device when every weight is
        # offloaded to disk; disk is a storage target, never a tensor device.
        return torch.device("cpu")
    raise RuntimeError(
        "The local language model has no available compute device for its input."
    )


def _generate(
    messages: list[dict[str, str]],
    max_new_tokens: int,
    on_progress: ProgressCallback | None = None,
    on_chunk: TextCallback | None = None,
    enable_thinking: bool = True,
) -> str:
    if on_progress is not None:
        on_progress("Preparing the response…")
    generator = _get_generator(on_progress)
    if on_progress is not None:
        on_progress("Generating response…")

    try:
        from transformers import TextIteratorStreamer
    except ImportError as error:
        raise RuntimeError(
            "Local AI dependencies are missing. Install the packages in "
            "requirements.txt and try again."
        ) from error

    model = generator.model
    tokenizer = generator.tokenizer
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(_input_device(model, torch))
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=1.0,
    )
    generation_error: list[BaseException] = []

    def generate() -> None:
        try:
            model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        except BaseException as error:
            generation_error.append(error)

    generation_thread = Thread(target=generate, daemon=True, name="local-ai-generation")
    generation_thread.start()
    generated_chunks: list[str] = []
    while True:
        try:
            chunk = next(streamer)
        except StopIteration:
            break
        except Empty:
            if not generation_thread.is_alive():
                break
            continue
        if chunk:
            generated_chunks.append(chunk)
            if on_chunk is not None:
                on_chunk(chunk)
    generation_thread.join()
    if generation_error:
        raise RuntimeError("The local language model failed during generation.") from (
            generation_error[0]
        )
    generated = "".join(generated_chunks).strip()
    if not generated:
        raise RuntimeError("The local language model returned an empty response.")
    return generated


def transcribe_audio(
    path: str | Path,
    on_progress: ProgressCallback | None = None,
    on_chunk: TextCallback | None = None,
) -> str:
    """Transcribe local audio and emit each completed segment as it is recognized."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Audio file not found: {target}")
    if on_progress is not None:
        on_progress("Preparing audio for transcription…")
    try:
        import librosa
    except ImportError as error:
        raise RuntimeError(
            "Audio decoding support is missing. Install the packages in "
            "requirements.txt and try again."
        ) from error
    audio, sample_rate = librosa.load(str(target), sr=16000, mono=True)
    if len(audio) == 0:
        raise ValueError("No speech could be transcribed from this audio file.")
    recognizer = _get_transcriber(on_progress)
    samples_per_chunk = TRANSCRIPTION_CHUNK_SECONDS * sample_rate
    chunk_count = (len(audio) + samples_per_chunk - 1) // samples_per_chunk
    segments: list[str] = []
    for index, offset in enumerate(range(0, len(audio), samples_per_chunk)):
        if on_progress is not None:
            on_progress(f"Transcribing audio segment {index + 1} of {chunk_count}…")
        result = recognizer(
            {"raw": audio[offset : offset + samples_per_chunk], "sampling_rate": sample_rate}
        )
        if not isinstance(result, dict):
            raise RuntimeError("The local speech model returned an invalid response.")
        text = result.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        elapsed_seconds = offset / sample_rate
        minutes, seconds = divmod(int(elapsed_seconds), 60)
        segment = f"[{minutes}:{seconds:02d}] {text.strip()}"
        segments.append(segment)
        if on_chunk is not None:
            on_chunk(segment + "\n")
    if not segments:
        raise ValueError("No speech could be transcribed from this audio file.")
    return "\n".join(segments)


def summarize_text(
    text: str,
    title: str = "document",
    on_progress: ProgressCallback | None = None,
    on_chunk: TextCallback | None = None,
) -> str:
    """Generate a concise, source-grounded summary with the local language model."""
    if not text.strip():
        raise ValueError("There is no readable text to summarize.")
    source_excerpt = text[:SUMMARY_INPUT_MAX_CHARS]
    if len(text) > SUMMARY_INPUT_MAX_CHARS:
        source_excerpt += "\n[Remaining source text omitted to keep generation fast.]"
    messages = [
        {
            "role": "system",
            "content": (
                "Summarize the supplied file in exactly one concise paragraph "
                "of 4-5 short sentences, about 40-70 words total. Prioritize "
                "quick readability: use plain language, direct wording, and "
                "only the most important points. Do not use a heading, bullets, "
                "preamble, or reasoning. Never invent facts or add information "
                "absent from the source."
            ),
        },
        {
            "role": "user",
            "content": f"File: {title}\n\nSource text:\n{source_excerpt}",
        },
    ]
    return _generate(
        messages,
        max_new_tokens=SUMMARY_MAX_NEW_TOKENS,
        on_progress=on_progress,
        on_chunk=on_chunk,
        enable_thinking=False,
    )


def summarize_document(
    path: str | Path,
    on_progress: ProgressCallback | None = None,
    on_chunk: TextCallback | None = None,
) -> str:
    target = Path(path)
    if on_progress is not None:
        on_progress(f"Reading {target.name} locally…")
    return summarize_text(
        read_document_text(target),
        target.name,
        on_progress=on_progress,
        on_chunk=on_chunk,
    )


def answer_question(
    question: str,
    context: str,
    on_progress: ProgressCallback | None = None,
    on_chunk: TextCallback | None = None,
) -> str:
    """Answer a compilation question using only the supplied local file context."""
    if not question.strip():
        raise ValueError("Enter a question about this compilation.")
    if not context.strip():
        raise ValueError(
            "This compilation has no readable documents or saved transcripts to "
            "use as context."
        )
    messages = [
        {
            "role": "system",
            "content": (
                "Answer questions about the user's compilation using only the "
                "provided context. Be clear and concise. If the context does not "
                "contain the answer, say that you cannot determine it from these "
                "files. Do not invent facts."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Compilation context:\n{context[:30000]}\n\n"
                f"Question: {question.strip()}"
            ),
        },
    ]
    return _generate(
        messages,
        max_new_tokens=256,
        on_progress=on_progress,
        on_chunk=on_chunk,
        enable_thinking=False,
    )
