# Summaraizer

A Python/Tkinter desktop prototype for organizing files into compilations,
previewing sample audio transcriptions, generating per-file sample summaries,
and chatting about a compilation. AI-generated content is mocked; no AI service
is used.

## Run

Use Python 3.10 or later. Install the UI and audio dependencies, then launch the
app:

```powershell
python -m pip install -r requirements.txt
python summaraizer_app.py
```

Keep `summaraizer_app.py` and `mock_ai.py` in the same directory. Run the GUI
through `summaraizer_app.py`; it imports the mock transcription and summary
functions from `mock_ai.py`.

The interface combines Tkinter's native event loop with a CustomTkinter
navigation sidebar, buttons, cards, and editors, plus a Pillow-rendered
empty-state illustration. Audio playback supports the common formats accepted
by pygame. The sidebar provides persistent workflow navigation and compilation
switching; scrollbars appear only when their content needs scrolling.
Compilation names, file paths, sample transcriptions, and summaries are saved
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
