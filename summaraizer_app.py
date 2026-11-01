from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from tkinter import (
    BOTH,
    Canvas,
    LEFT,
    RIGHT,
    VERTICAL,
    X,
    Y,
    Button,
    Entry,
    Frame,
    Label,
    Text,
    Tk,
    Toplevel,
    filedialog,
    messagebox,
    ttk,
)
from typing import Callable


APP_NAME = "Summaraizer"
DATA_PATH = Path.home() / ".summaraizer" / "compilations.json"

DOCUMENT_TYPES = {".pdf": "PDF", ".txt": "Text"}
AUDIO_TYPES = {".mp3": "Audio", ".wav": "Audio"}
SUPPORTED_TYPES = {**DOCUMENT_TYPES, **AUDIO_TYPES}

COLORS = {
    "background": "#F5F6F8",
    "surface": "#FFFFFF",
    "sidebar": "#17212B",
    "sidebar_muted": "#AAB5C0",
    "text": "#18212B",
    "muted": "#65717E",
    "border": "#E1E5EA",
    "accent": "#536DFE",
    "accent_hover": "#4058E8",
    "accent_light": "#EEF0FF",
    "success": "#18794E",
    "success_light": "#E8F5EE",
    "warning": "#946200",
    "warning_light": "#FFF4D6",
    "danger": "#B42318",
}


@dataclass
class Compilation:
    id: str
    name: str
    created_at: str
    files: list[str] = field(default_factory=list)
    transcriptions: dict[str, str] = field(default_factory=dict)
    accepted_transcriptions: list[str] = field(default_factory=list)
    summaries: dict[str, str] = field(default_factory=dict)
    master_summary: str = ""

    @classmethod
    def create(cls, name: str) -> Compilation:
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            created_at=datetime.now().isoformat(timespec="seconds"),
        )

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Compilation:
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            created_at=str(data["created_at"]),
            files=[str(path) for path in data.get("files", [])],
            transcriptions={
                str(path): str(text)
                for path, text in dict(data.get("transcriptions", {})).items()
            },
            accepted_transcriptions=[
                str(path) for path in data.get("accepted_transcriptions", [])
            ],
            summaries={
                str(path): str(text)
                for path, text in dict(data.get("summaries", {})).items()
            },
            master_summary=str(data.get("master_summary", "")),
        )

    @property
    def audio_files(self) -> list[str]:
        return [path for path in self.files if classify_file(path) == "Audio"]

    @property
    def summary_files(self) -> list[str]:
        return sorted(
            (path for path in self.files if Path(path).suffix.lower() in DOCUMENT_TYPES),
            key=lambda path: Path(path).name.casefold(),
        )


class CompilationStore:
    def __init__(self, path: Path = DATA_PATH) -> None:
        self.path = path

    def load(self) -> list[Compilation]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        if not isinstance(data, list):
            raise ValueError(f"Compilation data in {self.path} must be a JSON list.")
        return [Compilation.from_dict(item) for item in data]

    def save(self, compilations: list[Compilation]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            [asdict(compilation) for compilation in compilations],
            indent=2,
            ensure_ascii=False,
        )
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.path.parent,
                delete=False,
                suffix=".tmp",
            ) as temporary_file:
                temporary_file.write(serialized)
                temporary_path = Path(temporary_file.name)
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def summary_path(self, compilation: Compilation) -> Path:
        return self.path.parent / compilation.id / "Summary.txt"

    def write_summary(self, compilation: Compilation) -> Path:
        path = self.summary_path(compilation)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(compilation.master_summary, encoding="utf-8")
        return path


def classify_file(path: str | Path) -> str | None:
    return SUPPORTED_TYPES.get(Path(path).suffix.lower())


def transcript_path_for(audio_path: str | Path) -> Path:
    audio = Path(audio_path)
    return audio.with_name(f"{audio.stem}-transcript.txt")


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
    paragraph = (
        f"[0:00]Sample transcription for “{name}”.\n"
        "[0:30] This also demonstrates the timestamps.\n"
        "[1:00] I am taking very long pauses. That is why the timestamps are like this.\n"
        "[1:30]Good result."
    )
    return "\n\n".join(paragraph for _ in range(10)), False


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
        detail = (
            "Summary of the audio transcription. \n"
        )
    else:
        detail = f"Sample summary for {name}."
    return f"{name}: {detail} (Sample summary — AI is not connected.)"


def aggregate_summaries(compilation: Compilation) -> str:
    sections = [
        compilation.summaries[path]
        for path in compilation.summary_files
        if path in compilation.summaries
    ]
    return (
        f"Compilation summary: {compilation.name}\n\n"
        + "\n\n".join(sections)
        if sections
        else ""
    )


class ScrollableFrame(Frame):
    def __init__(self, master: Frame, **kwargs: object) -> None:
        super().__init__(master, **kwargs)
        self.canvas = Canvas(
            self, highlightthickness=0, background=COLORS["background"]
        )
        self.scrollbar = ttk.Scrollbar(self, orient=VERTICAL, command=self.canvas.yview)
        self.content = Frame(self.canvas, background=COLORS["background"])
        self.window_id = self.canvas.create_window(
            (0, 0), window=self.content, anchor="nw"
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
        self.scrollbar.pack(side=RIGHT, fill=Y)
        self.content.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            ),
        )
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(self.window_id, width=event.width),
        )
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event: object) -> None:
        delta = getattr(event, "delta", 0)
        if delta:
            self.canvas.yview_scroll(int(-delta / 120), "units")


class SummaraizerApp:
    def __init__(self, root: Tk, store: CompilationStore | None = None) -> None:
        self.root = root
        self.store = store or CompilationStore()
        self.compilations = self.store.load()
        self.active_compilation: Compilation | None = None
        self.active_page = "home"
        self.audio_index = 0
        self.summary_index = 0
        self.summary_editor: Text | None = None
        self.summary_editor_path: str | None = None
        self.summary_drafts: dict[str, str] = {}
        self.audio_status_label: Label | None = None
        self.transcription_drafts: dict[str, str] = {}
        self.chat_entry: Entry | None = None
        self.chat_messages: list[tuple[str, str]] = []
        self.retryable_chat_responses: set[int] = set()
        self.page_container: Frame
        self.page_title: Label

        self._configure_styles()
        self.root.title(APP_NAME)
        self.root.geometry("1180x760")
        self.root.minsize(940, 620)
        self.root.configure(background=COLORS["background"])
        self._build_shell()
        self.show_page("home")

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "TScrollbar",
            background=COLORS["background"],
            troughcolor=COLORS["background"],
            bordercolor=COLORS["background"],
            arrowcolor=COLORS["muted"],
        )
        style.configure(
            "TProgressbar",
            background=COLORS["accent"],
            troughcolor=COLORS["border"],
            bordercolor=COLORS["border"],
        )

    def _build_shell(self) -> None:
        shell = Frame(self.root, background=COLORS["background"])
        shell.pack(fill=BOTH, expand=True)
        main = Frame(shell, background=COLORS["background"])
        main.pack(fill=BOTH, expand=True)
        topbar = Frame(
            main,
            background=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        topbar.pack(fill=X)
        self.page_title = Label(
            topbar,
            text="Home",
            font=("Segoe UI", 16, "bold"),
            foreground=COLORS["text"],
            background=COLORS["surface"],
            padx=28,
            pady=19,
        )
        self.page_title.pack(side=LEFT)
        self.active_name = Label(
            topbar,
            text="No compilation selected",
            font=("Segoe UI", 9),
            foreground=COLORS["muted"],
            background=COLORS["surface"],
            padx=24,
        )
        self.active_name.pack(side=RIGHT)
        self.page_container = Frame(main, background=COLORS["background"])
        self.page_container.pack(fill=BOTH, expand=True)

    def _persist(self) -> None:
        self.store.save(self.compilations)

    def _set_active(self, compilation: Compilation) -> None:
        self.active_compilation = compilation
        self.active_name.configure(text=compilation.name)

    def _require_compilation(self) -> Compilation | None:
        if self.active_compilation is None:
            messagebox.showinfo(
                "Choose a compilation",
                "Open or create a compilation before using this page.",
                parent=self.root,
            )
            self.show_page("home")
            return None
        return self.active_compilation

    def show_page(self, page: str) -> None:
        previous_page = self.active_page
        if page == "summary" and previous_page != "summary":
            compilation = self.active_compilation
            self.summary_drafts = (
                dict(compilation.summaries) if compilation is not None else {}
            )
        if page == "transcription" and previous_page != "transcription":
            compilation = self.active_compilation
            if compilation is not None:
                self.transcription_drafts = dict(compilation.transcriptions)
        self.active_page = page
        for child in self.page_container.winfo_children():
            child.destroy()
        titles = {
            "home": "Home",
            "viewer": "Compilation viewer",
            "transcription": "Audio Transcriptions",
            "summary": "Summary generation",
            "chat": "Deep Dive",
        }
        self.page_title.configure(text=titles[page])
        renderers: dict[str, Callable[[], None]] = {
            "home": self._render_home,
            "viewer": self._render_viewer,
            "transcription": self._render_transcription,
            "summary": self._render_summary,
            "chat": self._render_chat,
        }
        renderers[page]()

    def _page_intro(self, title: str, subtitle: str) -> Frame:
        intro = Frame(self.page_container, background=COLORS["background"])
        intro.pack(fill=X, padx=34, pady=(28, 20))
        Label(
            intro,
            text=title,
            font=("Segoe UI", 22, "bold"),
            foreground=COLORS["text"],
            background=COLORS["background"],
        ).pack(anchor="w")
        Label(
            intro,
            text=subtitle,
            font=("Segoe UI", 10),
            foreground=COLORS["muted"],
            background=COLORS["background"],
        ).pack(anchor="w", pady=(5, 0))
        return intro

    def _button(
        self,
        parent: Frame,
        text: str,
        command: Callable[[], None],
        *,
        primary: bool = False,
        compact: bool = False,
    ) -> Button:
        button = Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 9, "bold" if primary else "normal"),
            relief="flat",
            borderwidth=0,
            padx=16 if not compact else 10,
            pady=10 if not compact else 6,
            cursor="hand2",
            background=COLORS["accent"] if primary else COLORS["surface"],
            foreground="#FFFFFF" if primary else COLORS["text"],
            activebackground=(
                COLORS["accent_hover"] if primary else COLORS["accent_light"]
            ),
            activeforeground="#FFFFFF" if primary else COLORS["text"],
            highlightbackground=COLORS["border"],
            highlightthickness=1 if not primary else 0,
        )
        return button

    def _card(self, parent: Frame, **kwargs: object) -> Frame:
        return Frame(
            parent,
            background=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
            padx=20,
            pady=18,
            **kwargs,
        )

    def _render_home(self) -> None:
        intro = self._page_intro(
            "Your workspace",
            "Create a compilation to bring documents and audio together.",
        )
        self._button(
            intro,
            "+  New compilation",
            self._open_create_dialog,
            primary=True,
        ).pack(side=RIGHT, anchor="e", pady=(0, 2))
        intro.pack_configure(fill=X)

        summary = Frame(self.page_container, background=COLORS["background"])
        summary.pack(fill=X, padx=34, pady=(0, 14))
        Label(
            summary,
            text=f"{len(self.compilations)} compilation"
            f"{'' if len(self.compilations) == 1 else 's'}",
            font=("Segoe UI", 12, "bold"),
            foreground=COLORS["text"],
            background=COLORS["background"],
        ).pack(side=LEFT)
        Label(
            summary,
            text="Saved on this device",
            font=("Segoe UI", 9),
            foreground=COLORS["muted"],
            background=COLORS["background"],
        ).pack(side=RIGHT)

        if not self.compilations:
            empty = self._card(self.page_container)
            empty.pack(fill=X, padx=34, pady=4)
            Label(
                empty,
                text="Welcome to Summaraizer",
                font=("Segoe UI", 14, "bold"),
                foreground=COLORS["text"],
                background=COLORS["surface"],
            ).pack(anchor="w")
            Label(
                empty,
                text=(
                    "This application uses AI to compile and summarize information "
                    "from text and audio files. To begin, create a new compilation."
                ),
                font=("Segoe UI", 10),
                foreground=COLORS["muted"],
                background=COLORS["surface"],
            ).pack(anchor="w", pady=(6, 14))
            self._button(
                empty,
                "Create your first compilation",
                self._open_create_dialog,
                primary=True,
            ).pack(anchor="w")
            return

        collection = ScrollableFrame(self.page_container)
        collection.pack(fill=BOTH, expand=True, padx=34, pady=(0, 24))
        for compilation in reversed(self.compilations):
            card = self._card(collection.content)
            card.pack(fill=X, pady=(0, 12))
            top = Frame(card, background=COLORS["surface"])
            top.pack(fill=X)
            Label(
                top,
                text=compilation.name,
                font=("Segoe UI", 13, "bold"),
                foreground=COLORS["text"],
                background=COLORS["surface"],
            ).pack(side=LEFT)
            self._button(
                top,
                "Open",
                lambda selected=compilation: self._open_compilation(selected),
                primary=True,
                compact=True,
            ).pack(side=RIGHT)
            self._button(
                top,
                "Delete",
                lambda selected=compilation: self._delete_compilation(selected),
                compact=True,
            ).pack(side=RIGHT, padx=(0, 6))
            self._button(
                top,
                "Rename",
                lambda selected=compilation: self._open_rename_dialog(selected),
                compact=True,
            ).pack(side=RIGHT, padx=(0, 6))
            date_label = compilation.created_at[:10]
            audio_count = len(compilation.audio_files)
            details = (
                f"{len(compilation.files)} files"
                f"  ·  {audio_count} audio"
                f"  ·  Created {date_label}"
            )
            Label(
                card,
                text=details,
                font=("Segoe UI", 9),
                foreground=COLORS["muted"],
                background=COLORS["surface"],
            ).pack(anchor="w", pady=(8, 0))
            if compilation.master_summary:
                Label(
                    card,
                    text="Summary finalized",
                    font=("Segoe UI", 9, "bold"),
                    foreground=COLORS["success"],
                    background=COLORS["surface"],
                ).pack(anchor="w", pady=(6, 0))

    def _open_rename_dialog(self, compilation: Compilation) -> None:
        dialog = Toplevel(self.root)
        dialog.title("Rename compilation")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.configure(background=COLORS["surface"])
        dialog.grab_set()
        content = Frame(dialog, background=COLORS["surface"], padx=24, pady=22)
        content.pack(fill=BOTH, expand=True)
        Label(
            content,
            text="Rename compilation",
            font=("Segoe UI", 15, "bold"),
            foreground=COLORS["text"],
            background=COLORS["surface"],
        ).pack(anchor="w")
        Label(
            content,
            text="Compilation name",
            font=("Segoe UI", 9, "bold"),
            foreground=COLORS["text"],
            background=COLORS["surface"],
        ).pack(anchor="w", pady=(14, 6))
        entry = Entry(
            content,
            font=("Segoe UI", 10),
            foreground=COLORS["text"],
            background=COLORS["surface"],
            insertbackground=COLORS["text"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["accent"],
        )
        entry.pack(fill=X, ipady=9)
        entry.insert(0, compilation.name)
        entry.selection_range(0, "end")
        actions = Frame(content, background=COLORS["surface"])
        actions.pack(fill=X, pady=(18, 0))

        def rename() -> None:
            name = entry.get().strip()
            if not name:
                messagebox.showwarning(
                    "Name required",
                    "Enter a name for this compilation.",
                    parent=dialog,
                )
                entry.focus_set()
                return
            compilation.name = name
            self._persist()
            if self.active_compilation is compilation:
                self._set_active(compilation)
            dialog.destroy()
            self.show_page("home")

        self._button(actions, "Cancel", dialog.destroy).pack(side=RIGHT)
        self._button(actions, "Save name", rename, primary=True).pack(
            side=RIGHT, padx=(0, 8)
        )
        dialog.bind("<Return>", lambda _event: rename())
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.update_idletasks()
        dialog.geometry(
            f"440x205+{self.root.winfo_rootx() + 180}+{self.root.winfo_rooty() + 150}"
        )
        dialog.after_idle(entry.focus_set)

    def _delete_compilation(self, compilation: Compilation) -> None:
        if not messagebox.askyesno(
            "Delete compilation?",
            f"Delete “{compilation.name}” and its saved summary? "
            "Original files will not be deleted.",
            icon="warning",
            parent=self.root,
        ):
            return
        self.compilations.remove(compilation)
        summary_path = self.store.summary_path(compilation)
        try:
            summary_path.unlink(missing_ok=True)
        except OSError as error:
            messagebox.showerror(
                "Unable to delete saved summary",
                f"The compilation was removed, but its saved summary could not be "
                f"deleted:\n{error}",
                parent=self.root,
            )
        self._persist()
        if self.active_compilation is compilation:
            self.active_compilation = None
            self.active_name.configure(text="No compilation selected")
            self.summary_drafts.clear()
            self.transcription_drafts.clear()
            self.chat_messages.clear()
            self.retryable_chat_responses.clear()
        self.show_page("home")

    def _open_create_dialog(self) -> None:
        dialog = Toplevel(self.root)
        dialog.title("New compilation")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.configure(background=COLORS["surface"])
        dialog.grab_set()
        content = Frame(dialog, background=COLORS["surface"], padx=24, pady=22)
        content.pack(fill=BOTH, expand=True)
        Label(
            content,
            text="Name your compilation",
            font=("Segoe UI", 15, "bold"),
            foreground=COLORS["text"],
            background=COLORS["surface"],
        ).pack(anchor="w")
        Label(
            content,
            text="You can add files after creating it.",
            font=("Segoe UI", 9),
            foreground=COLORS["muted"],
            background=COLORS["surface"],
        ).pack(anchor="w", pady=(5, 14))
        Label(
            content,
            text="Compilation name",
            font=("Segoe UI", 9, "bold"),
            foreground=COLORS["text"],
            background=COLORS["surface"],
        ).pack(anchor="w", pady=(0, 6))
        entry = Entry(
            content,
            font=("Segoe UI", 10),
            foreground=COLORS["text"],
            background=COLORS["surface"],
            insertbackground=COLORS["text"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["accent"],
        )
        entry.pack(fill=X, ipady=9)
        actions = Frame(content, background=COLORS["surface"])
        actions.pack(fill=X, pady=(18, 0))

        def create() -> None:
            name = entry.get().strip()
            if not name:
                messagebox.showwarning(
                    "Name required",
                    "Enter a name for this compilation.",
                    parent=dialog,
                )
                entry.focus_set()
                return
            compilation = Compilation.create(name)
            self.compilations.append(compilation)
            self._persist()
            self._set_active(compilation)
            dialog.destroy()
            self.show_page("viewer")

        self._button(actions, "Cancel", dialog.destroy).pack(side=RIGHT)
        self._button(actions, "Create", create, primary=True).pack(
            side=RIGHT, padx=(0, 8)
        )
        dialog.bind("<Return>", lambda _event: create())
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.update_idletasks()
        dialog.geometry(
            f"440x240+{self.root.winfo_rootx() + 180}+{self.root.winfo_rooty() + 150}"
        )
        dialog.after_idle(entry.focus_set)

    def _open_compilation(self, compilation: Compilation) -> None:
        self._set_active(compilation)
        self.show_page("viewer")

    def _choose_files(self) -> None:
        compilation = self._require_compilation()
        if compilation is None:
            return
        paths = filedialog.askopenfilenames(
            parent=self.root,
            title="Add files to compilation",
            filetypes=[
                ("Supported files", "*.txt *.pdf *.mp3 *.wav"),
                ("Text and PDF", "*.txt *.pdf"),
                ("Audio", "*.mp3 *.wav"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        invalid: list[str] = []
        added = 0
        existing_paths = {str(Path(path).resolve()) for path in compilation.files}
        for path in paths:
            if classify_file(path) is None:
                invalid.append(Path(path).name)
                continue
            resolved_path = str(Path(path).resolve())
            if resolved_path in existing_paths:
                continue
            compilation.files.append(resolved_path)
            existing_paths.add(resolved_path)
            added += 1
        if added:
            compilation.master_summary = ""
            self._persist()
        added_audio = any(
            classify_file(path) == "Audio"
            and str(Path(path).resolve()) in compilation.files
            for path in paths
        )
        if added_audio:
            self.audio_index = next(
                index
                for index, path in enumerate(compilation.audio_files)
                if Path(path).resolve()
                in {Path(selected).resolve() for selected in paths}
            )
            self.show_page("transcription")
        else:
            self.show_page("viewer")
        if invalid:
            messagebox.showwarning(
                "Unsupported file type",
                "These files were not added:\n"
                + "\n".join(invalid[:8])
                + ("\n…" if len(invalid) > 8 else "")
                + "\n\nOnly .txt, .pdf, .mp3, and .wav files are supported.",
                parent=self.root,
            )
        elif added == 0:
            messagebox.showinfo(
                "No new files",
                "All selected files were already in this compilation.",
                parent=self.root,
            )

    def _render_viewer(self) -> None:
        compilation = self._require_compilation()
        if compilation is None:
            return
        intro = self._page_intro(
            compilation.name,
            "Review your files, add more, then continue to transcription or summaries.",
        )
        self._button(
            intro, "+  Add files", self._choose_files, primary=True
        ).pack(side=RIGHT, anchor="e")
        panel = self._card(self.page_container)
        if compilation.master_summary:
            summary_card = self._card(self.page_container)
            summary_card.pack(fill=X, padx=34, pady=(0, 14))
            summary_header = Frame(summary_card, background=COLORS["surface"])
            summary_header.pack(fill=X, pady=(0, 8))
            Label(
                summary_header,
                text="Summary.txt",
                font=("Segoe UI", 12, "bold"),
                foreground=COLORS["text"],
                background=COLORS["surface"],
            ).pack(side=LEFT)
            summary_path = self.store.summary_path(compilation)
            Label(
                summary_header,
                text=str(summary_path),
                font=("Segoe UI", 8),
                foreground=COLORS["muted"],
                background=COLORS["surface"],
            ).pack(side=LEFT, padx=(10, 0))
            summary_text = (
                summary_path.read_text(encoding="utf-8")
                if summary_path.is_file()
                else compilation.master_summary
            )
            summary_area = Frame(summary_card, background=COLORS["surface"])
            summary_area.pack(fill=X)
            summary_content = Text(
                summary_area,
                height=7,
                wrap="word",
                font=("Segoe UI", 9),
                foreground=COLORS["text"],
                background="#FAFBFC",
                relief="flat",
                padx=10,
                pady=8,
            )
            summary_scroll = ttk.Scrollbar(
                summary_area,
                orient=VERTICAL,
                command=summary_content.yview,
            )
            summary_content.configure(yscrollcommand=summary_scroll.set)
            summary_content.pack(side=LEFT, fill=X, expand=True)
            summary_scroll.pack(side=RIGHT, fill=Y)
            summary_content.insert("1.0", summary_text)
            summary_content.configure(state="disabled")
        panel.pack(fill=BOTH, expand=True, padx=34, pady=(0, 18))
        header = Frame(panel, background=COLORS["surface"])
        header.pack(fill=X, pady=(0, 12))
        Label(
            header,
            text="Files in this compilation",
            font=("Segoe UI", 12, "bold"),
            foreground=COLORS["text"],
            background=COLORS["surface"],
        ).pack(side=LEFT)
        Label(
            header,
            text=f"{len(compilation.files)} files",
            font=("Segoe UI", 9),
            foreground=COLORS["muted"],
            background=COLORS["surface"],
        ).pack(side=RIGHT)

        if not compilation.files:
            Label(
                panel,
                text="Add documents or audio to get started.",
                font=("Segoe UI", 10),
                foreground=COLORS["muted"],
                background=COLORS["surface"],
                pady=22,
            ).pack(anchor="w")
        else:
            file_list = ScrollableFrame(panel)
            file_list.pack(fill=BOTH, expand=True)
            for path in compilation.files:
                row = Frame(
                    file_list.content,
                    background=COLORS["surface"],
                    highlightbackground=COLORS["border"],
                    highlightthickness=1,
                    padx=12,
                    pady=10,
                )
                row.pack(fill=X, pady=(0, 8))
                file_type = classify_file(path) or "Unsupported"
                Label(
                    row,
                    text="♪" if file_type == "Audio" else "▤",
                    font=("Segoe UI", 15, "bold"),
                    foreground=COLORS["accent"],
                    background=COLORS["accent_light"],
                    width=3,
                    pady=7,
                ).pack(side=LEFT, padx=(0, 12))
                detail = Frame(row, background=COLORS["surface"])
                detail.pack(side=LEFT, fill=X, expand=True)
                Label(
                    detail,
                    text=Path(path).name,
                    font=("Segoe UI", 10, "bold"),
                    foreground=COLORS["text"],
                    background=COLORS["surface"],
                    anchor="w",
                ).pack(fill=X)
                Label(
                    detail,
                    text=f"{file_type}  ·  {Path(path).suffix.lower()}",
                    font=("Segoe UI", 9),
                    foreground=COLORS["muted"],
                    background=COLORS["surface"],
                    anchor="w",
                ).pack(fill=X, pady=(3, 0))
                self._button(
                    row,
                    "Remove",
                    lambda selected=path: self._remove_file(selected),
                    compact=True,
                ).pack(side=RIGHT)

        actions = Frame(self.page_container, background=COLORS["background"])
        actions.pack(fill=X, padx=34, pady=(0, 24))
        self._button(
            actions, "Return to Home", lambda: self.show_page("home")
        ).pack(
            side=LEFT
        )
        self._button(
            actions,
            "Summary Generation",
            lambda: self.show_page("summary"),
            primary=True,
        ).pack(side=RIGHT, padx=(0, 8))
        if compilation.audio_files:
            self._button(
                actions,
                "Audio Transcriptions",
                self._continue_from_viewer,
            ).pack(side=RIGHT, padx=(0, 8))
        self._button(
            actions,
            "Deep Dive",
            lambda: self.show_page("chat"),
        ).pack(side=RIGHT, padx=(0, 8))
        if compilation.master_summary:
            self._button(
                actions,
                "View summary",
                self._show_master_summary,
            ).pack(side=RIGHT, padx=(0, 8))
            self._button(
                actions,
                "Export summary",
                self._export_master_summary,
            ).pack(side=RIGHT, padx=(0, 8))

    def _remove_file(self, path: str) -> None:
        compilation = self._require_compilation()
        if compilation is None:
            return
        compilation.files.remove(path)
        compilation.transcriptions.pop(path, None)
        if path in compilation.accepted_transcriptions:
            compilation.accepted_transcriptions.remove(path)
        compilation.summaries.pop(path, None)
        compilation.master_summary = ""
        self._persist()
        self.show_page("viewer")

    def _continue_from_viewer(self) -> None:
        compilation = self._require_compilation()
        if compilation is None:
            return
        if compilation.audio_files:
            self.audio_index = 0
            self.show_page("transcription")
        else:
            self.summary_index = 0
            self.show_page("summary")

    def _render_transcription(self) -> None:
        compilation = self._require_compilation()
        if compilation is None:
            return
        audio_files = compilation.audio_files
        if not audio_files:
            self._page_intro(
                "No audio files",
                "Add an audio file in the compilation viewer to use transcription.",
            )
            self._button(
                self.page_container,
                "Back to compilation",
                lambda: self.show_page("viewer"),
            ).pack(anchor="w", padx=34)
            return
        self.audio_index = min(self.audio_index, len(audio_files) - 1)
        self._page_intro(
            "Audio transcription",
            "Review a sample transcript for each audio file. AI transcription is not connected.",
        )
        body = Frame(self.page_container, background=COLORS["background"])
        body.pack(fill=BOTH, expand=True, padx=34, pady=(0, 24))
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        files_panel = self._card(body, width=245)
        files_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        files_panel.grid_propagate(False)
        Label(
            files_panel,
            text="Audio files",
            font=("Segoe UI", 11, "bold"),
            foreground=COLORS["text"],
            background=COLORS["surface"],
        ).pack(anchor="w", pady=(0, 10))
        for index, path in enumerate(audio_files):
            self._button(
                files_panel,
                f"{'●  ' if index == self.audio_index else ''}{Path(path).name}",
                lambda selected=index: self._select_audio(selected),
                primary=index == self.audio_index,
            ).pack(fill=X, pady=(0, 7))

        content = Frame(body, background=COLORS["background"])
        content.grid(row=0, column=1, sticky="nsew")
        path = audio_files[self.audio_index]
        transcript = self.transcription_drafts.get(
            path, compilation.transcriptions.get(path, "")
        )
        audio_card = self._card(content)
        audio_card.pack(fill=X, pady=(0, 12))
        audio_header = Frame(audio_card, background=COLORS["surface"])
        audio_header.pack(fill=X)
        file_info = Frame(audio_header, background=COLORS["surface"])
        file_info.pack(side=LEFT, fill=X, expand=True)
        Label(
            file_info,
            text=Path(path).name,
            font=("Segoe UI", 11, "bold"),
            foreground=COLORS["text"],
            background=COLORS["surface"],
        ).pack(anchor="w")
        confidence = "Not generated"
        if transcript:
            confidence = (
                "5%"
                if transcript.startswith(
                    ("[Low confidence]", "No speech could be detected")
                )
                else "96%"
            )
        self.audio_status_label = Label(
            file_info,
            text=f"Confidence: {confidence}  ·  Ready to play",
            font=("Segoe UI", 9),
            foreground=COLORS["muted"],
            background=COLORS["surface"],
        )
        self.audio_status_label.pack(anchor="w", pady=(4, 0))
        player_controls = Frame(audio_header, background=COLORS["surface"])
        player_controls.pack(side=LEFT, padx=(10, 8))
        self._button(
            player_controls,
            "▶  Play",
            lambda selected=path: self._play_audio(selected),
            primary=True,
            compact=True,
        ).pack(side=LEFT)
        self._button(
            player_controls,
            "Ⅱ  Pause",
            self._pause_audio,
            compact=True,
        ).pack(side=LEFT, padx=8)
        self._button(
            player_controls, "■  Stop", self._stop_audio, compact=True
        ).pack(side=LEFT)
        actions = Frame(audio_header, background=COLORS["surface"])
        actions.pack(side=RIGHT, padx=(8, 0))
        self._button(
            actions,
            "Generate",
            lambda selected=path: self._generate_transcription(selected),
            primary=True,
            compact=True,
        ).pack(side=RIGHT, padx=(6, 0))
        self._button(
            actions,
            "Save",
            lambda selected=path: self._save_transcription(selected),
            compact=True,
        ).pack(side=RIGHT, padx=(6, 0))
        self._button(
            actions,
            "Cancel",
            lambda selected=path: self._discard_transcription(selected),
            compact=True,
        ).pack(side=RIGHT)

        transcript_card = self._card(content)
        transcript_card.pack(fill=BOTH, expand=True)
        transcript_header = Frame(transcript_card, background=COLORS["surface"])
        transcript_header.pack(fill=X, pady=(0, 10))
        Label(
            transcript_header,
            text="Transcript",
            font=("Segoe UI", 12, "bold"),
            foreground=COLORS["text"],
            background=COLORS["surface"],
        ).pack(side=LEFT)
        low_confidence = transcript.startswith("[Low confidence]") or transcript.startswith(
            "No speech could be detected"
        )
        if transcript:
            badge_text = "Low confidence" if low_confidence else "Sample result"
            badge_color = COLORS["warning"] if low_confidence else COLORS["success"]
            Label(
                transcript_header,
                text=badge_text,
                font=("Segoe UI", 9, "bold"),
                foreground=badge_color,
                background=(
                    COLORS["warning_light"] if low_confidence else COLORS["success_light"]
                ),
                padx=8,
                pady=4,
            ).pack(side=RIGHT)
        transcript_area = Frame(transcript_card, background=COLORS["surface"])
        transcript_area.pack(fill=BOTH, expand=True)
        transcript_widget = Text(
            transcript_area,
            wrap="word",
            font=("Segoe UI", 10),
            foreground=COLORS["text"],
            background="#FAFBFC",
            relief="flat",
            padx=12,
            pady=12,
        )
        transcript_scroll = ttk.Scrollbar(
            transcript_area, orient=VERTICAL, command=transcript_widget.yview
        )
        transcript_widget.configure(yscrollcommand=transcript_scroll.set)
        transcript_widget.pack(side=LEFT, fill=BOTH, expand=True)
        transcript_scroll.pack(side=RIGHT, fill=Y)
        transcript_widget.insert("1.0", transcript or "Generate a sample transcription.")
        transcript_widget.configure(state="disabled")

    def _select_audio(self, index: int) -> None:
        self._stop_audio()
        self.audio_index = index
        self.show_page("transcription")

    def _generate_transcription(self, path: str) -> None:
        compilation = self._require_compilation()
        if compilation is None:
            return
        transcript, low_confidence = mock_transcription(path)
        self.transcription_drafts[path] = transcript
        self.show_page("transcription")
        if low_confidence:
            messagebox.showwarning(
                "Low-confidence sample",
                "This filename produces a low-confidence or no-speech sample state.",
                parent=self.root,
            )

    def _save_transcription(self, path: str) -> None:
        compilation = self._require_compilation()
        if compilation is None:
            return
        transcript = self.transcription_drafts.get(
            path, compilation.transcriptions.get(path)
        )
        if not transcript:
            messagebox.showinfo(
                "No transcript to save",
                "Generate a transcript before saving it.",
                parent=self.root,
            )
            return
        transcript_path = transcript_path_for(path)
        try:
            transcript_path.write_text(transcript, encoding="utf-8")
        except OSError as error:
            messagebox.showerror(
                "Unable to save transcript",
                f"Could not write {transcript_path}:\n{error}",
                parent=self.root,
            )
            return
        compilation.transcriptions[path] = transcript
        if str(transcript_path) not in compilation.files:
            compilation.files.append(str(transcript_path))
        if path not in compilation.accepted_transcriptions:
            compilation.accepted_transcriptions.append(path)
        compilation.master_summary = ""
        self.transcription_drafts.pop(path, None)
        self._persist()
        self.show_page("viewer")

    def _discard_transcription(self, path: str) -> None:
        self.transcription_drafts.pop(path, None)
        self._stop_audio()
        self.show_page("viewer")

    def _play_audio(self, path: str) -> None:
        try:
            import pygame
        except ImportError:
            messagebox.showerror(
                "Audio playback unavailable",
                "Install the audio dependency with: pip install -r requirements.txt",
                parent=self.root,
            )
            return
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init()
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            if self.audio_status_label is not None:
                status = self.audio_status_label.cget("text")
                self.audio_status_label.configure(
                    text=status.replace("Ready to play", "Playing")
                )
        except pygame.error as error:
            messagebox.showerror(
                "Unable to play audio",
                f"Could not play {Path(path).name}:\n{error}",
                parent=self.root,
            )

    def _pause_audio(self) -> None:
        try:
            import pygame
        except ImportError:
            messagebox.showerror(
                "Audio playback unavailable",
                "Install the audio dependency with: pip install -r requirements.txt",
                parent=self.root,
            )
            return
        if pygame.mixer.get_init() is not None and pygame.mixer.music.get_busy():
            pygame.mixer.music.pause()
            if self.audio_status_label is not None:
                self.audio_status_label.configure(
                    text=self.audio_status_label.cget("text").replace(
                        "Playing", "Paused"
                    )
                )
        elif pygame.mixer.get_init() is not None:
            pygame.mixer.music.unpause()
            if self.audio_status_label is not None:
                self.audio_status_label.configure(
                    text=self.audio_status_label.cget("text").replace(
                        "Paused", "Playing"
                    )
                )

    def _stop_audio(self) -> None:
        try:
            import pygame
        except ImportError:
            return
        if pygame.mixer.get_init() is not None:
            pygame.mixer.music.stop()
            if self.audio_status_label is not None:
                self.audio_status_label.configure(
                    text=self.audio_status_label.cget("text")
                    .replace("Playing", "Ready to play")
                    .replace("Paused", "Ready to play")
                )

    def _render_summary(self) -> None:
        compilation = self._require_compilation()
        if compilation is None:
            return
        files = compilation.summary_files
        if not files:
            self._page_intro(
                "No text or PDF files to summarize",
                "Summary generation only includes .txt and .pdf files in this compilation.",
            )
            self._button(
                self.page_container,
                "Cancel",
                lambda: self.show_page("viewer"),
            ).pack(anchor="w", padx=34)
            return
        self.summary_index = min(self.summary_index, len(files) - 1)
        self._page_intro(
            "Summary generation",
            "Generate a sample summary for each file, then finalize a compilation summary.",
        )
        body = Frame(self.page_container, background=COLORS["background"])
        body.pack(fill=BOTH, expand=True, padx=34, pady=(0, 16))
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        files_panel = self._card(body, width=285)
        files_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        files_panel.grid_propagate(False)
        Label(
            files_panel,
            text="Compilation files",
            font=("Segoe UI", 11, "bold"),
            foreground=COLORS["text"],
            background=COLORS["surface"],
        ).pack(anchor="w", pady=(0, 10))
        for index, path in enumerate(files):
            done = path in self.summary_drafts
            label = f"{'✓' if done else '○'}  {Path(path).name}"
            self._button(
                files_panel,
                label,
                lambda selected=index: self._select_summary_file(selected),
                primary=index == self.summary_index,
            ).pack(fill=X, pady=(0, 7))

        content = Frame(body, background=COLORS["background"])
        content.grid(row=0, column=1, sticky="nsew")
        path = files[self.summary_index]
        summary_card = self._card(content)
        summary_card.pack(fill=BOTH, expand=True)
        Label(
            summary_card,
            text=Path(path).name,
            font=("Segoe UI", 12, "bold"),
            foreground=COLORS["text"],
            background=COLORS["surface"],
        ).pack(anchor="w", pady=(0, 12))
        summary_text = self.summary_drafts.get(path, "")
        if not summary_text:
            Label(
                summary_card,
                text="Generate a sample summary or enter your own below.",
                font=("Segoe UI", 9),
                foreground=COLORS["muted"],
                background=COLORS["surface"],
            ).pack(anchor="w", pady=(0, 8))
        summary_area = Frame(summary_card, background=COLORS["surface"])
        summary_area.pack(fill=BOTH, expand=True)
        summary_widget = Text(
            summary_area,
            wrap="word",
            font=("Segoe UI", 10),
            foreground=COLORS["text"],
            background="#FAFBFC",
            relief="solid",
            borderwidth=1,
            padx=12,
            pady=12,
        )
        summary_scroll = ttk.Scrollbar(
            summary_area, orient=VERTICAL, command=summary_widget.yview
        )
        summary_widget.configure(yscrollcommand=summary_scroll.set)
        summary_widget.pack(side=LEFT, fill=BOTH, expand=True)
        summary_scroll.pack(side=RIGHT, fill=Y)
        summary_widget.insert("1.0", summary_text)
        self.summary_editor = summary_widget
        self.summary_editor_path = path
        controls = Frame(content, background=COLORS["background"])
        controls.pack(fill=X, pady=(12, 0))
        self._button(
            controls,
            "Generate summary for file",
            lambda selected=path: self._generate_summary(selected),
            primary=True,
        ).pack(side=RIGHT)
        self._button(
            controls,
            "Save edits",
            self._save_summary_edits,
        ).pack(side=RIGHT, padx=(0, 8))
        self._button(
            controls,
            "Previous",
            lambda: self._select_summary_file(max(0, self.summary_index - 1)),
        ).pack(side=LEFT)
        self._button(
            controls,
            "Next",
            lambda: self._select_summary_file(
                min(len(files) - 1, self.summary_index + 1)
            ),
        ).pack(side=LEFT, padx=(8, 0))

        footer = Frame(self.page_container, background=COLORS["background"])
        footer.pack(fill=X, padx=34, pady=(0, 22))
        self._button(
            footer,
            "Cancel",
            self._cancel_summary_edits,
        ).pack(side=LEFT)
        if compilation.master_summary:
            self._button(
                footer,
                "View master summary",
                self._show_master_summary,
            ).pack(side=RIGHT, padx=(0, 8))
            self._button(
                footer,
                "Export master summary",
                self._export_master_summary,
            ).pack(side=RIGHT)
            Label(
                footer,
                text="Master summary finalized",
                font=("Segoe UI", 9, "bold"),
                foreground=COLORS["success"],
                background=COLORS["background"],
            ).pack(side=RIGHT, padx=(0, 12))
        self._button(
            footer,
            "Finalize compilation",
            self._finalize_summary,
            primary=True,
        ).pack(side=RIGHT, padx=(0, 10))

    def _select_summary_file(self, index: int) -> None:
        compilation = self._require_compilation()
        if compilation is None or not compilation.summary_files:
            return
        self._stage_current_summary()
        self.summary_index = max(0, min(index, len(compilation.summary_files) - 1))
        self.show_page("summary")

    def _stage_current_summary(self) -> None:
        editor = self.summary_editor
        path = self.summary_editor_path
        if (
            editor is None
            or path is None
            or not editor.winfo_exists()
        ):
            return
        text = editor.get("1.0", "end-1c").strip()
        if text:
            self.summary_drafts[path] = text

    def _save_summary_edits(self) -> None:
        compilation = self._require_compilation()
        if compilation is None:
            return
        self._stage_current_summary()
        compilation.summaries = dict(self.summary_drafts)
        compilation.master_summary = ""
        self._persist()
        self.summary_drafts.clear()
        self.show_page("viewer")

    def _cancel_summary_edits(self) -> None:
        self.summary_drafts.clear()
        self.show_page("viewer")

    def _generate_summary(self, path: str) -> None:
        if self._require_compilation() is None:
            return
        self._stage_current_summary()
        self.summary_drafts[path] = mock_summary(
            path,
            self.active_compilation.transcriptions.get(path, "")
            if self.active_compilation is not None
            else "",
        )
        self.show_page("summary")

    def _finalize_summary(self) -> None:
        compilation = self._require_compilation()
        if compilation is None:
            return
        self._stage_current_summary()
        missing = [
            Path(path).name
            for path in compilation.summary_files
            if not self.summary_drafts.get(path, "").strip()
        ]
        if missing:
            messagebox.showinfo(
                "Summaries still needed",
                "Generate an individual summary for each file before finalizing:\n"
                + "\n".join(missing[:8])
                + ("\n…" if len(missing) > 8 else ""),
                parent=self.root,
            )
            first_missing = next(
                index
                for index, path in enumerate(compilation.summary_files)
                if not self.summary_drafts.get(path, "").strip()
            )
            self._select_summary_file(first_missing)
            return
        compilation.summaries = dict(self.summary_drafts)
        compilation.master_summary = aggregate_summaries(compilation)
        try:
            self.store.write_summary(compilation)
            self._persist()
        except OSError as error:
            messagebox.showerror(
                "Unable to finalize summary",
                f"Could not save Summary.txt:\n{error}",
                parent=self.root,
            )
            return
        self.summary_drafts.clear()
        self.show_page("viewer")

    def _show_master_summary(self) -> None:
        compilation = self._require_compilation()
        if compilation is None or not compilation.master_summary:
            return
        dialog = Toplevel(self.root)
        dialog.title(f"{compilation.name} — Master summary")
        dialog.geometry("720x540")
        dialog.transient(self.root)
        dialog.configure(background=COLORS["background"])
        content = Frame(dialog, background=COLORS["background"], padx=20, pady=18)
        content.pack(fill=BOTH, expand=True)
        Label(
            content,
            text="Master summary",
            font=("Segoe UI", 16, "bold"),
            foreground=COLORS["text"],
            background=COLORS["background"],
        ).pack(anchor="w", pady=(0, 12))
        text_area = Frame(content, background=COLORS["surface"])
        text_area.pack(fill=BOTH, expand=True)
        summary_text = Text(
            text_area,
            wrap="word",
            font=("Segoe UI", 10),
            foreground=COLORS["text"],
            background=COLORS["surface"],
            relief="flat",
            padx=14,
            pady=14,
        )
        scrollbar = ttk.Scrollbar(
            text_area, orient=VERTICAL, command=summary_text.yview
        )
        summary_text.configure(yscrollcommand=scrollbar.set)
        summary_text.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)
        summary_text.insert("1.0", compilation.master_summary)
        summary_text.configure(state="disabled")
        actions = Frame(content, background=COLORS["background"])
        actions.pack(fill=X, pady=(12, 0))
        self._button(
            actions,
            "Close",
            dialog.destroy,
        ).pack(side=RIGHT)
        self._button(
            actions,
            "Export text file",
            self._export_master_summary,
            primary=True,
        ).pack(side=RIGHT, padx=(0, 8))

    def _export_master_summary(self) -> None:
        compilation = self._require_compilation()
        if compilation is None or not compilation.master_summary:
            return
        destination = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export master summary",
            initialfile=f"{compilation.name}_summary.txt",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not destination:
            return
        Path(destination).write_text(compilation.master_summary, encoding="utf-8")
        messagebox.showinfo(
            "Summary exported",
            f"Master summary saved to:\n{destination}",
            parent=self.root,
        )

    def _render_chat(self) -> None:
        if self._require_compilation() is None:
            return
        self._page_intro(
            "Deep Dive",
            "Ask a question about this compilation. Retrieval-augmented AI is not connected.",
        )
        card = self._card(self.page_container)
        card.pack(fill=BOTH, expand=True, padx=34, pady=(0, 22))
        conversation = ScrollableFrame(card)
        conversation.pack(fill=BOTH, expand=True)
        if not self.chat_messages:
            Label(
                conversation.content,
                text="Ask a question to start exploring your compilation.",
                font=("Segoe UI", 10),
                foreground=COLORS["muted"],
                background=COLORS["background"],
                pady=18,
            ).pack(anchor="w")
        for message_index, (speaker, text) in enumerate(self.chat_messages):
            message = Frame(
                conversation.content,
                background=COLORS["accent_light"]
                if speaker == "You"
                else "#F1F3F5",
                padx=12,
                pady=10,
            )
            message.pack(
                anchor="e" if speaker == "You" else "w",
                padx=(48, 4) if speaker == "You" else (4, 48),
                pady=(0, 10),
            )
            Label(
                message,
                text=speaker,
                font=("Segoe UI", 8, "bold"),
                foreground=COLORS["accent"] if speaker == "You" else COLORS["muted"],
                background=message.cget("background"),
            ).pack(anchor="w")
            Label(
                message,
                text=text,
                font=("Segoe UI", 10),
                foreground=COLORS["text"],
                background=message.cget("background"),
                justify=LEFT,
                wraplength=650,
            ).pack(anchor="w", pady=(3, 0))
            if message_index in self.retryable_chat_responses:
                self._button(
                    message,
                    "Retry",
                    lambda selected=message_index: self._retry_chat_response(
                        selected
                    ),
                    compact=True,
                ).pack(anchor="w", pady=(8, 0))
        composer = Frame(card, background=COLORS["surface"])
        composer.pack(fill=X, pady=(14, 0))
        entry = Entry(
            composer,
            font=("Segoe UI", 10),
            relief="solid",
            borderwidth=1,
        )
        self.chat_entry = entry
        entry.pack(side=LEFT, fill=X, expand=True, ipady=9, padx=(0, 10))

        self._button(
            composer, "Send", self._send_chat_message, primary=True
        ).pack(side=RIGHT)
        self._button(
            composer, "Close", self._close_chat
        ).pack(side=RIGHT, padx=(0, 8))
        entry.bind("<Return>", lambda _event: self._send_chat_message())
        entry.focus_set()

    def _send_chat_message(self) -> None:
        question = self.chat_entry.get().strip() if self.chat_entry else ""
        if not question:
            return
        self.chat_messages.extend(
            [
                ("You", question),
                ("Summaraizer", "The answer is banana; moon is a kind of sandwich."),
            ]
        )
        self.retryable_chat_responses.add(len(self.chat_messages) - 1)
        self.show_page("chat")

    def _retry_chat_response(self, response_index: int) -> None:
        if response_index not in self.retryable_chat_responses:
            return
        if not 0 <= response_index < len(self.chat_messages):
            return
        speaker, _poor_response = self.chat_messages[response_index]
        if speaker != "Summaraizer":
            return
        self.chat_messages[response_index] = (
            speaker,
            "The first 10 digits of pi are 3.141592653.",
        )
        self.retryable_chat_responses.remove(response_index)
        self.show_page("chat")

    def _close_chat(self) -> None:
        self.chat_entry = None
        self.show_page("viewer")


def main() -> None:
    root = Tk()
    try:
        SummaraizerApp(root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        messagebox.showerror(
            "Unable to load Summaraizer",
            f"Could not load saved compilations from {DATA_PATH}:\n{error}",
            parent=root,
        )
        root.destroy()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
