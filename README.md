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
Files can be added via the "+ Add files" button or by dragging and dropping them
directly into the compilation viewer page. Files and finalized summaries can be
opened directly from the compilation viewer or double-clicked to view.

Adding audio opens transcription automatically with sample transcripts generated
immediately on open. Transcripts are directly editable, and the primary action
is "Save Edits", which writes a sibling file such as `recording-transcript.txt`;
Cancel discards unsaved transcription changes.

Only `.txt` and `.pdf` files, including saved transcripts, appear in summary
generation. Mockup summaries are generated automatically upon opening rather than
waiting for a generate command. During summary review, a split-screen format
allows comparing the source document with the generated summary side by side.
The primary button advances with "Next File"; when reaching the final file in the
compilation, the "Save" button is removed and the primary button turns into
"Finalize" to aggregate and complete the compilation summary. Transcription filenames
containing `silent`, `empty`, or `no-speech` show the no-speech state; names
containing `unclear`, `noise`, or `low-confidence` show a low-confidence sample.
