# Summaraizer

A Python/Tkinter desktop prototype for organizing files into compilations,
previewing sample audio transcriptions, generating per-file sample summaries,
and chatting about a compilation. AI-generated content is mocked; no AI service
is used.

## Run

Use Python 3.10 or later. Install the audio playback dependency, then launch the
app:

```powershell
python -m pip install -r requirements.txt
python summaraizer_app.py
```

The GUI uses Tkinter, which is included with most Python distributions. Audio
playback supports the common formats accepted by pygame. Compilation names,
file paths, sample transcriptions, and summaries are saved locally in
`~/.summaraizer/compilations.json`. Finalized compilations also save
`Summary.txt` under their app data folder, and the viewer displays its contents.

Supported input file extensions match the design: `.txt` and `.pdf` documents,
plus `.mp3` and `.wav` audio. Files are referenced in place rather than copied.
Adding audio opens transcription automatically. Saving a transcript writes a
sibling file such as `recording-transcript.txt`; Cancel discards unsaved
transcription changes. Only `.txt` and `.pdf` files, including saved
transcripts, appear in summary generation. Summaries follow filename order and
can be edited, saved, or discarded before finalizing. Transcription filenames
containing `silent`, `empty`, or `no-speech` show the no-speech state; names
containing `unclear`, `noise`, or `low-confidence` show a low-confidence sample.
