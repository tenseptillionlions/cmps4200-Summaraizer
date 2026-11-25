# Summaraizer

A Python/Tkinter desktop prototype for organizing files into compilations,
transcribing audio, generating per-file summaries, and chatting about a
compilation. AI inference runs locally on the user's device; no hosted inference
service is used.

## Run

Use Python 3.10 or later. Install the UI and local AI dependencies, then launch
the app:

```powershell
python -m pip install -r requirements.txt
python summaraizer_app.py
```

Run the GUI through `summaraizer_app.py`. AI features are implemented in
`ai_backend.py` and select models on first use based on available hardware.
Devices with sufficient free VRAM use `openai/whisper-medium` and
`Qwen/Qwen3-4B-Instruct-2507`; devices with less VRAM use the smaller Whisper
and Qwen3-0.6B models on GPU when they fit, or run those small models entirely
on CPU. In particular, the CPU path avoids mapping the large Qwen model to disk,
which can make generation extremely slow. Model weights are downloaded and
cached locally the first time each model is needed, so first use requires an
internet connection and can take time. The 4B Qwen model may require Hugging
Face access approval depending on its availability.

The larger CUDA models are memory-intensive. CPU-specific models are smaller to
avoid slow disk offloading. MP3 transcription also requires FFmpeg to be
installed and available on `PATH`.

The interface combines Tkinter's native event loop with a CustomTkinter
navigation sidebar, buttons, cards, and editors, plus a Pillow-rendered
empty-state illustration. Audio playback supports the common formats accepted
by pygame. The sidebar provides persistent workflow navigation and compilation
switching; scrollbars appear only when their content needs scrolling.
Compilation names, file paths, transcriptions, and summaries are saved
locally in `~/.summaraizer/compilations.json`. Finalized compilations also save
`Summary.txt` under their app data folder, and the viewer displays its contents.

Supported input file extensions match the design: `.txt` and `.pdf` documents,
plus `.mp3` and `.wav` audio. Files are referenced in place rather than copied.
Files can be added via the "+ Add files" button or by dragging and dropping them
directly into the compilation viewer page. Files and finalized summaries can be
opened directly from the compilation viewer or double-clicked to view.
On Windows, file drops are registered on the app window and its child controls
so drops from File Explorer work across the viewer surface; the drop area also
opens the file picker when clicked.

Adding audio opens the transcription screen. Choose "Transcribe" to run Whisper
locally. The app shows model-loading and segment progress as recognized
transcription appears in the editor. The result is directly editable, and "Save
Edits" writes a sibling file such as `recording-transcript.txt`. Cancel
discards unsaved transcription changes.

Only `.txt` and `.pdf` files, including saved transcripts, appear in summary
generation. The selected file's local summary starts generating when the summary
screen opens, with model-loading status and generated text streaming into the
editor; summaries can also be regenerated on demand. PDFs are read locally for
their text content. During summary review, a split-screen format allows
comparing the source document with the generated summary side by side.
Summary Generation uses one concise paragraph of 4–5 short sentences, in plain
language for quick readability, with any selected model. It limits output to
200 tokens and caps long source text at 12,000 characters to reduce generation
time.
The primary button advances with "Next File"; when reaching the final file in the
compilation, the "Save" button is removed and the primary button turns into
"Finalize" to aggregate and complete the compilation summary. Transcripts and
summaries are editable before being saved. Deep Dive splits local file text and
saved transcripts into passages, ranks them against each question with a local
BM25 retriever, and sends the most relevant passages to the on-device language
model. Responses stream into the conversation and remain grounded in the
retrieved source material.
