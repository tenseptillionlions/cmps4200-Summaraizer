from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
import customtkinter as ctk
from PIL import Image, ImageDraw
from tkinter import (
    BOTH,
    Canvas,
    LEFT,
    RIGHT,
    VERTICAL,
    X,
    Y,
    Entry,
    Frame,
    Label,
    Text,
    Tk,
    TclError,
    Toplevel,
    filedialog,
    messagebox,
)
from typing import Callable

from mock_ai import mock_summary, mock_transcription


APP_NAME = "Summaraizer"
DATA_PATH = Path.home() / ".summaraizer" / "compilations.json"

DOCUMENT_TYPES = {".pdf": "PDF", ".txt": "Text"}
AUDIO_TYPES = {".mp3": "Audio", ".wav": "Audio"}
SUPPORTED_TYPES = {**DOCUMENT_TYPES, **AUDIO_TYPES}

COLORS = {
    "background": "#F8FAFC",
    "surface": "#FFFFFF",
    "sidebar": "#17212B",
    "sidebar_muted": "#AAB5C0",
    "text": "#0F172A",
    "body": "#475569",
    "muted": "#64748B",
    "border": "#E2E8F0",
    "accent": "#2563EB",
    "accent_hover": "#1D4ED8",
    "accent_light": "#EFF6FF",
    "sidebar_active": "#2A3A4B",
    "success": "#10B981",
    "success_hover": "#059669",
    "success_light": "#ECFDF5",
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


def display_filename(path: str | Path, limit: int = 42) -> str:
    name = Path(path).name
    if len(name) <= limit:
        return name
    return f"{name[: limit // 2 - 2]}...{name[-(limit // 2 - 1):]}"


def transcript_path_for(audio_path: str | Path) -> Path:
    audio = Path(audio_path)
    return audio.with_name(f"{audio.stem}-transcript.txt")


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
    def __init__(
        self, master: Frame, *, background: str = COLORS["background"], **kwargs: object
    ) -> None:
        super().__init__(master, **kwargs)
        self.canvas = Canvas(
            self, highlightthickness=0, background=background
        )
        self.scrollbar = AutoHideScrollbar(
            self,
            orient=VERTICAL,
            command=self.canvas.yview,
            dark=background == COLORS["sidebar"],
        )
        self.content = Frame(self.canvas, background=background)
        self.window_id = self.canvas.create_window(
            (0, 0), window=self.content, anchor="nw"
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
        self.scrollbar.pack(side=RIGHT, fill=Y)
        self.content.bind(
            "<Configure>",
            self._update_scroll_region,
        )
        self.canvas.bind(
            "<Configure>",
            self._on_canvas_configure,
        )
        self.bind_all("<MouseWheel>", self._on_mousewheel, add="+")

    def _update_scroll_region(self, _event: object = None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: object) -> None:
        self.canvas.itemconfigure(
            self.window_id, width=getattr(event, "width", self.canvas.winfo_width())
        )
        self._update_scroll_region()

    def _on_mousewheel(self, event: object) -> None:
        widget = getattr(event, "widget", None)
        inside_content = False
        while widget is not None:
            if widget is self.content:
                inside_content = True
                break
            widget = getattr(widget, "master", None)

        if not inside_content and getattr(event, "widget", None) not in (
            self.canvas,
            self.scrollbar,
        ):
            return

        delta = getattr(event, "delta", 0)
        if delta:
            units = int(-delta / 120) or (-1 if delta > 0 else 1)
            self.canvas.yview_scroll(units, "units")


class AutoHideScrollbar(ctk.CTkScrollbar):
    def __init__(
        self,
        master: Frame,
        *,
        orient: str = VERTICAL,
        command: Callable[..., None],
        dark: bool = False,
    ) -> None:
        super().__init__(
            master,
            orientation="vertical" if orient == VERTICAL else "horizontal",
            command=command,
            width=9,
            fg_color=COLORS["sidebar"] if dark else "#F1F5F9",
            button_color="#475569" if dark else "#CBD5E1",
            button_hover_color="#64748B" if dark else "#94A3B8",
        )

    def set(self, first: str, last: str) -> None:
        if float(first) <= 0 and float(last) >= 1:
            if self.winfo_manager():
                self.pack_forget()
        elif not self.winfo_manager():
            self.pack(side=RIGHT, fill=Y)
        super().set(first, last)


class TranscriptTextbox(ctk.CTkTextbox):
    def cget(self, attribute_name: str) -> object:
        if attribute_name == "state":
            return self._textbox.cget("state")
        return super().cget(attribute_name)


class SummaraizerApp:
    def __init__(self, root: Tk, store: CompilationStore | None = None) -> None:
        ctk.set_appearance_mode("light")
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
        self.summary_progress: ctk.CTkProgressBar | None = None
        self.summary_stream_after: str | None = None
        self.summary_stream_generation = 0
        self.summary_streaming_path: str | None = None
        self.transcription_editor: Text | TranscriptTextbox | None = None
        self.transcription_editor_path: str | None = None
        self.audio_status_label: Label | None = None
        self.audio_slider: ctk.CTkSlider | None = None
        self.audio_time_label: Label | None = None
        self.audio_duration_seconds = 0.0
        self.audio_position_after: str | None = None
        self.transcription_drafts: dict[str, str] = {}
        self._drop_window_procs: dict[int, tuple[int, object]] = {}
        self.chat_entry: ctk.CTkTextbox | None = None
        self.chat_messages: list[tuple[str, str]] = []
        self.retryable_chat_responses: set[int] = set()
        self.page_container: ctk.CTkFrame
        self.sidebar_nav: ctk.CTkFrame
        self.sidebar_compilations: ScrollableFrame
        self.drag_drop_enabled = False
        self.empty_state_icon: ctk.CTkImage | None = None

        self.root.title(APP_NAME)
        self.root.geometry("1280x820")
        self.root.minsize(1024, 700)
        self.root.configure(background=COLORS["background"])
        self._build_shell()
        self._setup_drag_and_drop()
        if self.drag_drop_enabled:
            self.root.bind("<Map>", self._schedule_drop_targets, add="+")
        self.show_page("home")

    def _build_shell(self) -> None:
        shell = ctk.CTkFrame(
            self.root, fg_color=COLORS["background"], corner_radius=0
        )
        shell.pack(fill=BOTH, expand=True)
        sidebar = ctk.CTkFrame(
            shell,
            width=216,
            fg_color=COLORS["sidebar"],
            corner_radius=0,
        )
        sidebar.pack(side=LEFT, fill=Y)
        sidebar.pack_propagate(False)
        ctk.CTkLabel(
            sidebar,
            text="SUMMARAIZER",
            font=("Segoe UI", 15, "bold"),
            text_color="#FFFFFF",
            anchor="w",
        ).pack(fill=X, padx=20, pady=(24, 20))
        self._button(
            sidebar,
            "+  New compilation",
            self._open_create_dialog,
            primary=True,
        ).pack(fill=X, padx=14, pady=(0, 20))

        self.sidebar_nav = ctk.CTkFrame(sidebar, fg_color="transparent")
        self.sidebar_nav.pack(fill=X, padx=10)
        ctk.CTkLabel(
            sidebar,
            text="COMPILATIONS",
            font=("Segoe UI", 10, "bold"),
            text_color=COLORS["sidebar_muted"],
            anchor="w",
        ).pack(fill=X, padx=20, pady=(22, 8))
        self.sidebar_compilations = ScrollableFrame(
            sidebar,
            background=COLORS["sidebar"],
            borderwidth=0,
            highlightthickness=0,
        )
        self.sidebar_compilations.pack(
            fill=BOTH, expand=True, padx=10, pady=(0, 8)
        )
        ctk.CTkLabel(
            sidebar,
            text="Saved on this device",
            font=("Segoe UI", 9),
            text_color=COLORS["sidebar_muted"],
            anchor="w",
        ).pack(side="bottom", fill=X, padx=20, pady=18)

        main = ctk.CTkFrame(shell, fg_color=COLORS["background"], corner_radius=0)
        main.pack(side=LEFT, fill=BOTH, expand=True)
        self.page_container = ctk.CTkFrame(
            main, fg_color=COLORS["background"], corner_radius=0
        )
        self.page_container.pack(fill=BOTH, expand=True)
        self._refresh_sidebar()

    def _refresh_sidebar(self) -> None:
        for child in self.sidebar_nav.winfo_children():
            child.destroy()
        for child in self.sidebar_compilations.content.winfo_children():
            child.destroy()
        pages = [
            ("Home", "home"),
            ("Files", "viewer"),
            ("Transcription", "transcription"),
            ("Summary", "summary"),
            ("Deep Dive", "chat"),
        ]
        for label, page in pages:
            available = page == "home" or self.active_compilation is not None
            active = self.active_page == page
            button = ctk.CTkButton(
                self.sidebar_nav,
                text=label,
                command=lambda selected=page: self.show_page(selected),
                height=40,
                corner_radius=8,
                anchor="w",
                font=("Segoe UI", 12, "bold" if active else "normal"),
                fg_color=COLORS["sidebar_active"] if active else "transparent",
                hover_color="#34495D",
                text_color="#FFFFFF" if available else COLORS["sidebar_muted"],
                state="normal" if available else "disabled",
            )
            button.pack(fill=X, pady=2)

        for compilation in reversed(self.compilations):
            active = self.active_compilation is compilation
            name = compilation.name
            if len(name) > 16:
                name = f"{name[:13]}..."
            button = ctk.CTkButton(
                self.sidebar_compilations.content,
                text=f"{name}  ·  {len(compilation.files)}",
                command=lambda selected=compilation: self._open_compilation(selected),
                height=38,
                corner_radius=8,
                anchor="w",
                font=("Segoe UI", 11, "bold" if active else "normal"),
                fg_color=COLORS["sidebar_active"] if active else "transparent",
                hover_color="#34495D",
                text_color="#FFFFFF",
            )
            button.pack(fill=X, pady=2)

    def _setup_drag_and_drop(self) -> None:
        if sys.platform == "win32":
            try:
                self._setup_native_drag_drop()
                self.drag_drop_enabled = True
                self._schedule_drop_targets()
            except (AttributeError, OSError) as error:
                self.drag_drop_enabled = False
                print(f"File drag and drop could not be enabled: {error}", file=sys.stderr)

    def _setup_native_drag_drop(self) -> None:
        import ctypes
        from ctypes import wintypes

        self._user32 = ctypes.windll.user32
        self._shell32 = ctypes.windll.shell32

        WNDPROC = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t,
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        )
        self._WNDPROC = WNDPROC
        self._GWLP_WNDPROC = -4
        self._WM_DROPFILES = 0x0233
        self._set_window_long_ptr = getattr(
            self._user32,
            "SetWindowLongPtrW",
            getattr(self._user32, "SetWindowLongW", None),
        )
        if not self._set_window_long_ptr:
            raise OSError("SetWindowLongPtr is unavailable")
        self._set_window_long_ptr.argtypes = [
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._set_window_long_ptr.restype = ctypes.c_void_p
        self._call_window_proc = self._user32.CallWindowProcW
        self._call_window_proc.argtypes = [
            ctypes.c_void_p,
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        self._call_window_proc.restype = ctypes.c_ssize_t
        self._shell32.DragAcceptFiles.argtypes = [wintypes.HWND, wintypes.BOOL]
        self._shell32.DragAcceptFiles.restype = None
        self._shell32.DragQueryFileW.argtypes = [
            wintypes.WPARAM,
            wintypes.UINT,
            wintypes.LPWSTR,
            wintypes.UINT,
        ]
        self._shell32.DragQueryFileW.restype = wintypes.UINT
        self._shell32.DragFinish.argtypes = [wintypes.WPARAM]
        self._shell32.DragFinish.restype = None
        self.root.bind("<Destroy>", lambda _e: self._restore_drop_targets(), add="+")

    def _register_drop_targets(self) -> None:
        if not self.drag_drop_enabled or not hasattr(self, "_WNDPROC"):
            return
        if not self.root.winfo_viewable():
            return
        self.root.update_idletasks()
        widgets = [self.root]
        while widgets:
            widget = widgets.pop()
            try:
                hwnd = int(widget.winfo_id())
                if hwnd and hwnd not in self._drop_window_procs:
                    self._hook_drop_window(hwnd)
                widgets.extend(widget.winfo_children())
            except (AttributeError, OSError, TclError):
                continue

    def _schedule_drop_targets(self, _event: object = None) -> None:
        if self.drag_drop_enabled and self.root.winfo_viewable():
            self.root.after_idle(self._register_drop_targets)

    def _hook_drop_window(self, hwnd: int) -> None:
        import ctypes

        self._shell32.DragAcceptFiles(hwnd, True)

        def wnd_proc(h_wnd: int, msg: int, w_param: int, l_param: int) -> int:
            if msg == self._WM_DROPFILES:
                paths: list[str] = []
                try:
                    count = self._shell32.DragQueryFileW(w_param, 0xFFFFFFFF, None, 0)
                    for index in range(count):
                        length = self._shell32.DragQueryFileW(
                            w_param, index, None, 0
                        )
                        buffer = ctypes.create_unicode_buffer(length + 1)
                        self._shell32.DragQueryFileW(
                            w_param, index, buffer, length + 1
                        )
                        paths.append(buffer.value)
                finally:
                    self._shell32.DragFinish(w_param)
                self.root.after(0, lambda: self._on_files_dropped(paths))
                return 0
            return self._call_window_proc(
                old_window_proc, h_wnd, msg, w_param, l_param
            )

        callback = self._WNDPROC(wnd_proc)
        old_window_proc = self._set_window_long_ptr(
            hwnd,
            self._GWLP_WNDPROC,
            ctypes.cast(callback, ctypes.c_void_p),
        )
        if not old_window_proc:
            raise OSError(f"Could not register file drop target for window {hwnd}")
        self._drop_window_procs[hwnd] = (old_window_proc, callback)

    def _restore_drop_targets(self) -> None:
        if not self._drop_window_procs:
            return
        for hwnd, (old_window_proc, _callback) in self._drop_window_procs.items():
            try:
                self._set_window_long_ptr(
                    hwnd, self._GWLP_WNDPROC, old_window_proc
                )
                self._shell32.DragAcceptFiles(hwnd, False)
            except (AttributeError, OSError):
                continue
        self._drop_window_procs.clear()

    def _on_files_dropped(self, files: list[str]) -> None:
        if self.active_page != "viewer" or self.active_compilation is None:
            return
        self._add_file_paths(files)

    def _persist(self) -> None:
        self.store.save(self.compilations)

    def _set_active(self, compilation: Compilation) -> None:
        self.active_compilation = compilation

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
        self.summary_stream_generation += 1
        if self.summary_stream_after is not None:
            self.root.after_cancel(self.summary_stream_after)
            self.summary_stream_after = None
        if self.summary_progress is not None:
            self.summary_progress.stop()
            self.summary_progress = None
        if self.audio_position_after is not None:
            self.root.after_cancel(self.audio_position_after)
            self.audio_position_after = None
        if self.drag_drop_enabled:
            self._restore_drop_targets()
        self.summary_streaming_path = None
        if page == "summary":
            compilation = self.active_compilation
            if compilation is not None:
                if previous_page != "summary":
                    self.summary_drafts = dict(compilation.summaries)
                for s_path in compilation.summary_files:
                    if (
                        s_path not in self.summary_drafts
                        or not self.summary_drafts[s_path].strip()
                    ):
                        transcription = compilation.transcriptions.get(s_path, "")
                        self.summary_drafts[s_path] = mock_summary(
                            s_path, transcription
                        )
        if page == "transcription":
            compilation = self.active_compilation
            if compilation is not None:
                if previous_page != "transcription":
                    self.transcription_drafts = dict(compilation.transcriptions)
                for a_path in compilation.audio_files:
                    if (
                        a_path not in self.transcription_drafts
                        or not self.transcription_drafts[a_path].strip()
                    ):
                        sample, _ = mock_transcription(a_path)
                        self.transcription_drafts[a_path] = sample
        self.active_page = page
        for child in self.page_container.winfo_children():
            child.destroy()
        renderers: dict[str, Callable[[], None]] = {
            "home": self._render_home,
            "viewer": self._render_viewer,
            "transcription": self._render_transcription,
            "summary": self._render_summary,
            "chat": self._render_chat,
        }
        renderers[page]()
        self._refresh_sidebar()
        if self.drag_drop_enabled:
            self._schedule_drop_targets()

    def _page_intro(self, title: str, subtitle: str) -> ctk.CTkFrame:
        intro = ctk.CTkFrame(
            self.page_container, fg_color=COLORS["background"], corner_radius=0
        )
        intro.pack(fill=X, padx=24, pady=(24, 18))
        ctk.CTkLabel(
            intro,
            text=title,
            font=("Segoe UI", 23, "bold"),
            text_color=COLORS["text"],
            anchor="w",
        ).pack(anchor="w")
        ctk.CTkLabel(
            intro,
            text=subtitle,
            font=("Segoe UI", 11),
            text_color=COLORS["muted"],
            anchor="w",
        ).pack(anchor="w", pady=(5, 0))
        return intro

    def _button(
        self,
        parent: Frame,
        text: str,
        command: Callable[[], None],
        *,
        primary: bool = False,
        success: bool = False,
        compact: bool = False,
        large: bool = False,
        padx: int | None = None,
        pady: int | None = None,
        font_size: int | None = None,
    ) -> ctk.CTkButton:
        if padx is None:
            padx = 22 if large else (10 if compact else 16)
        if pady is None:
            pady = 12 if large else (6 if compact else 10)
        if font_size is None:
            font_size = 13 if large else (9 if compact else 10)
        height = 56 if large else 2 * pady + 24
        width = max(70, min(280, round(len(text) * font_size * 0.62 + 2 * padx)))
        fg_color = (
            COLORS["success"] if success else
            COLORS["accent"] if primary else COLORS["surface"]
        )
        hover_color = (
            COLORS["success_hover"] if success else
            COLORS["accent_hover"] if primary else COLORS["accent_light"]
        )
        button = ctk.CTkButton(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", font_size, "bold" if (primary or success or large) else "normal"),
            width=max(width, 300 if large else 0),
            height=height,
            corner_radius=8,
            border_width=0 if primary or success else 1,
            border_color=COLORS["border"],
            fg_color=fg_color,
            text_color="#FFFFFF" if (primary or success) else COLORS["text"],
            hover_color=hover_color,
            cursor="hand2",
        )
        return button

    def _card(self, parent: Frame, **kwargs: object) -> Frame:
        return ctk.CTkFrame(
            parent,
            fg_color=COLORS["surface"],
            border_color=COLORS["border"],
            border_width=1,
            corner_radius=10,
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
        summary.pack(fill=X, padx=24, pady=(0, 14))
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
            font=("Segoe UI", 10),
            foreground=COLORS["muted"],
            background=COLORS["background"],
        ).pack(side=RIGHT)

        if not self.compilations:
            empty = self._card(self.page_container)
            empty.pack(fill=X, padx=24, pady=4)
            empty_content = Frame(
                empty, background=COLORS["surface"], padx=28, pady=24
            )
            empty_content.pack(fill=X)
            empty_content.pack_propagate(False)
            empty_content.configure(height=300)
            empty_content.grid_columnconfigure(0, weight=1)
            icon_art = Image.new("RGBA", (72, 72), (0, 0, 0, 0))
            icon_draw = ImageDraw.Draw(icon_art)
            icon_draw.rounded_rectangle(
                (4, 4, 68, 68), radius=20, fill="#EEF2FF"
            )
            icon_draw.rounded_rectangle(
                (17, 24, 55, 52), radius=6, fill="#3B82F6"
            )
            icon_draw.rounded_rectangle((19, 17, 37, 27), radius=4, fill="#3B82F6")
            icon_draw.line((25, 34, 47, 34), fill="#FFFFFF", width=3)
            icon_draw.line((25, 42, 42, 42), fill="#BFDBFE", width=3)
            icon_draw.ellipse((45, 43, 62, 60), fill="#FFFFFF")
            icon_draw.line((53, 47, 53, 56), fill="#3B82F6", width=2)
            icon_draw.line((49, 51, 57, 51), fill="#3B82F6", width=2)
            self.empty_state_icon = ctk.CTkImage(
                light_image=icon_art, dark_image=icon_art, size=(72, 72)
            )
            ctk.CTkLabel(
                empty_content, image=self.empty_state_icon, text=""
            ).pack(pady=(0, 10))
            Label(
                empty_content,
                text="Welcome to Summaraizer",
                font=("Segoe UI", 20, "bold"),
                foreground=COLORS["text"],
                background=COLORS["surface"],
            ).pack(anchor="center")
            Label(
                empty_content,
                text=(
                    "This application uses AI to compile and summarize information "
                    "from text and audio files. To begin, create a new compilation."
                ),
                font=("Segoe UI", 11),
                foreground=COLORS["body"],
                background=COLORS["surface"],
                justify="center",
                wraplength=540,
            ).pack(anchor="center", pady=(6, 14))
            self._button(
                empty_content,
                "+  Create your first compilation",
                self._open_create_dialog,
                primary=True,
                large=True,
                font_size=14,
            ).pack(anchor="center")
            return

        collection = ScrollableFrame(self.page_container)
        collection.pack(fill=BOTH, expand=True, padx=24, pady=(0, 20))
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
        if paths:
            self._add_file_paths(list(paths))

    def _add_file_paths(self, paths: list[str | Path]) -> None:
        compilation = self._require_compilation()
        if compilation is None or not paths:
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

    def _open_file(self, path: str | Path) -> None:
        target = Path(path).resolve()
        if not target.exists():
            messagebox.showerror(
                "File not found",
                f"The file does not exist on disk:\n{target}",
                parent=self.root,
            )
            return
        opened = False
        try:
            if hasattr(os, "startfile"):
                os.startfile(str(target))
                opened = True
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
                opened = True
            else:
                subprocess.Popen(["xdg-open", str(target)])
                opened = True
        except Exception:
            opened = False

        if not opened:
            if target.suffix.lower() == ".txt":
                self._preview_text_file(target)
            else:
                messagebox.showerror(
                    "Unable to open file",
                    f"Could not open {target.name} with default application.",
                    parent=self.root,
                )

    def _preview_text_file(self, target: Path) -> None:
        dialog = Toplevel(self.root)
        dialog.title(target.name)
        dialog.geometry("640x480")
        dialog.transient(self.root)
        dialog.configure(background=COLORS["background"])

        content = Frame(dialog, background=COLORS["background"], padx=20, pady=18)
        content.pack(fill=BOTH, expand=True)

        Label(
            content,
            text=target.name,
            font=("Segoe UI", 14, "bold"),
            foreground=COLORS["text"],
            background=COLORS["background"],
        ).pack(anchor="w", pady=(0, 10))

        text_area = Frame(content, background=COLORS["surface"])
        text_area.pack(fill=BOTH, expand=True)

        preview = Text(
            text_area,
            wrap="word",
            font=("Segoe UI", 10),
            foreground=COLORS["text"],
            background=COLORS["surface"],
            relief="flat",
            padx=12,
            pady=12,
        )
        scrollbar = AutoHideScrollbar(
            text_area, orient=VERTICAL, command=preview.yview
        )
        preview.configure(yscrollcommand=scrollbar.set)
        preview.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

        try:
            content_text = target.read_text(encoding="utf-8")
        except Exception:
            content_text = target.read_text(encoding="latin-1", errors="replace")

        preview.insert("1.0", content_text)
        preview.configure(state="disabled")

        actions = Frame(content, background=COLORS["background"])
        actions.pack(fill=X, pady=(12, 0))
        self._button(actions, "Close", dialog.destroy).pack(side=RIGHT)

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

        actions = ctk.CTkFrame(
            self.page_container, fg_color=COLORS["background"], corner_radius=0
        )
        actions.pack(side="bottom", fill=X, padx=24, pady=(8, 16))

        has_transcripts = bool(compilation.audio_files) or any(
            f.endswith("-transcript.txt") for f in compilation.files
        )

        if compilation.master_summary:
            self._button(
                actions,
                "Deep Dive",
                lambda: self.show_page("chat"),
                primary=True,
            ).pack(side=RIGHT, padx=(8, 0))
            self._button(
                actions,
                "Summary Generation",
                lambda: self.show_page("summary"),
            ).pack(side=RIGHT)
        else:
            self._button(
                actions,
                "Summary Generation",
                lambda: self.show_page("summary"),
                primary=True,
            ).pack(side=RIGHT)

            self._button(
                actions,
                "Deep Dive",
                lambda: self.show_page("chat"),
            ).pack(side=RIGHT, padx=(0, 8))
        if has_transcripts:
            self._button(
                actions,
                (
                    "Transcript viewing"
                    if compilation.master_summary or not compilation.audio_files
                    else "Audio Transcriptions"
                ),
                self._continue_from_viewer,
            ).pack(side=RIGHT, padx=(0, 8))

        # Top summary card if finalized
        if compilation.master_summary:
            summary_card = self._card(self.page_container)
            summary_card.pack(side="top", fill=X, padx=24, pady=(0, 12))
            summary_content_frame = Frame(
                summary_card, background=COLORS["surface"], padx=14, pady=12
            )
            summary_content_frame.pack(fill=X)
            summary_header = Frame(
                summary_content_frame, background=COLORS["surface"]
            )
            summary_header.pack(fill=X)
            Label(
                summary_header,
                text="Summary.txt",
                font=("Segoe UI", 13, "bold"),
                foreground=COLORS["text"],
                background=COLORS["surface"],
            ).pack(side=LEFT)
            summary_path = self.store.summary_path(compilation)
            self._button(
                summary_header,
                "Open",
                lambda p=summary_path: self._open_file(p),
                compact=True,
            ).pack(side=RIGHT, padx=(8, 0))
            self._button(
                summary_header,
                "Export",
                self._export_master_summary,
                compact=True,
            ).pack(side=RIGHT)
            Label(
                summary_content_frame,
                text="Finalized compilation summary",
                font=("Segoe UI", 9),
                foreground=COLORS["muted"],
                background=COLORS["surface"],
                anchor="w",
            ).pack(fill=X, pady=(3, 8))
            summary_text = (
                summary_path.read_text(encoding="utf-8")
                if summary_path.is_file()
                else compilation.master_summary
            )
            summary_area = Frame(
                summary_content_frame, background=COLORS["surface"]
            )
            summary_area.pack(fill=X)
            summary_content = Text(
                summary_area,
                height=5,
                wrap="word",
                font=("Segoe UI", 11),
                foreground=COLORS["text"],
                background="#FAFBFC",
                relief="flat",
                padx=12,
                pady=10,
            )
            summary_scroll = AutoHideScrollbar(
                summary_area,
                orient=VERTICAL,
                command=summary_content.yview,
            )
            summary_content.configure(yscrollcommand=summary_scroll.set)
            summary_content.pack(side=LEFT, fill=X, expand=True)
            summary_scroll.pack(side=RIGHT, fill=Y)
            summary_content.insert("1.0", summary_text)
            summary_content.configure(state="disabled")

        # Files panel takes remaining space in the middle
        panel = self._card(self.page_container)
        panel.pack(side="top", fill=BOTH, expand=True, padx=24, pady=(0, 8))
        panel_content = Frame(panel, background=COLORS["surface"], padx=14, pady=12)
        panel_content.pack(fill=BOTH, expand=True)
        header = Frame(panel_content, background=COLORS["surface"])
        header.pack(fill=X, pady=(0, 12))
        Label(
            header,
            text="Files in this compilation",
            font=("Segoe UI", 13, "bold"),
            foreground=COLORS["text"],
            background=COLORS["surface"],
        ).pack(side=LEFT)
        Label(
            header,
            text=f"{len(compilation.files)} files",
            font=("Segoe UI", 10),
            foreground=COLORS["muted"],
            background=COLORS["surface"],
        ).pack(side=RIGHT)

        drop_zone = ctk.CTkFrame(
            panel_content,
            fg_color="#EEF2FF",
            border_color="#C7D2FE",
            border_width=1,
            corner_radius=10,
        )
        drop_zone.pack(fill=X, pady=(0, 12), padx=2)
        Label(
            drop_zone,
            text="Drag and drop files here to add them to this compilation",
            font=("Segoe UI", 10, "bold"),
            foreground=COLORS["accent"],
            background="#EEF2FF",
            pady=12,
        ).pack(anchor="center")
        Label(
            drop_zone,
            text=(
                "Supported: .txt, .pdf, .mp3, .wav  ·  Click to browse"
                if self.drag_drop_enabled
                else "Drag and drop unavailable  ·  Click to browse files"
            ),
            font=("Segoe UI", 9),
            foreground=COLORS["muted"],
            background="#EEF2FF",
        ).pack(anchor="center", pady=(2, 0))
        drop_zone.bind("<Button-1>", lambda _e: self._choose_files())
        for child in drop_zone.winfo_children():
            child.bind("<Button-1>", lambda _e: self._choose_files())

        if not compilation.files:
            Label(
                panel_content,
                text="Add documents or audio to get started.",
                font=("Segoe UI", 11),
                foreground=COLORS["muted"],
                background=COLORS["surface"],
                pady=22,
            ).pack(anchor="w")
        else:
            file_list = ScrollableFrame(panel_content)
            file_list.pack(fill=BOTH, expand=True)
            for path in compilation.files:
                row = ctk.CTkFrame(
                    file_list.content,
                    fg_color=COLORS["surface"],
                    border_color=COLORS["border"],
                    border_width=1,
                    corner_radius=8,
                )
                row.pack(fill=X, pady=(0, 8))
                file_type = classify_file(path) or "Unsupported"
                extension = Path(path).suffix.lower().lstrip(".").upper()
                badge_colors = {
                    "PDF": ("#FEF2F2", "#B42318"),
                    "TXT": ("#EEF2FF", "#4338CA"),
                    "MP3": ("#F5F3FF", "#6D28D9"),
                    "WAV": ("#F5F3FF", "#6D28D9"),
                }
                badge_background, badge_foreground = badge_colors.get(
                    extension, (COLORS["accent_light"], COLORS["accent"])
                )
                ctk.CTkLabel(
                    row,
                    text=extension,
                    font=("Segoe UI", 9, "bold"),
                    text_color=badge_foreground,
                    fg_color=badge_background,
                    corner_radius=7,
                    width=44,
                    height=34,
                ).pack(side=LEFT, padx=(0, 12))
                detail = Frame(row, background=COLORS["surface"])
                detail.pack(side=LEFT, fill=X, expand=True)
                display_name = Path(path).name
                if len(display_name) > 56:
                    display_name = f"{display_name[:25]}...{display_name[-27:]}"
                Label(
                    detail,
                    text=display_name,
                    font=("Segoe UI", 11, "bold"),
                    foreground=COLORS["text"],
                    background=COLORS["surface"],
                    anchor="w",
                ).pack(fill=X)
                Label(
                    detail,
                    text=f"{file_type}  ·  {Path(path).suffix.lower()}",
                    font=("Segoe UI", 10),
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
                self._button(
                    row,
                    "Open",
                    lambda selected=path: self._open_file(selected),
                    compact=True,
                ).pack(side=RIGHT, padx=(0, 6))

                row.bind("<Double-Button-1>", lambda _e, p=path: self._open_file(p))
                detail.bind("<Double-Button-1>", lambda _e, p=path: self._open_file(p))
                for w in detail.winfo_children():
                    w.bind("<Double-Button-1>", lambda _e, p=path: self._open_file(p))

    def _remove_file(self, path: str) -> None:
        compilation = self._require_compilation()
        if compilation is None:
            return
        compilation.files.remove(path)
        compilation.transcriptions.pop(path, None)
        if path in compilation.accepted_transcriptions:
            compilation.accepted_transcriptions.remove(path)
        compilation.summaries.pop(path, None)
        self.summary_drafts.pop(path, None)
        self.transcription_drafts.pop(path, None)
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
        body.pack(fill=BOTH, expand=True, padx=24, pady=(0, 20))
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        files_panel = self._card(body, width=230)
        files_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        files_panel.grid_propagate(False)
        Label(
            files_panel,
            text="Audio files",
            font=("Segoe UI", 11, "bold"),
            foreground=COLORS["text"],
            background=COLORS["surface"],
        ).pack(anchor="w", pady=(0, 10))
        for index, a_path in enumerate(audio_files):
            self._button(
                files_panel,
                display_filename(a_path, 34),
                lambda selected=index: self._select_audio(selected),
                primary=index == self.audio_index,
            ).pack(fill=X, pady=(0, 7))

        content = Frame(body, background=COLORS["background"])
        content.grid(row=0, column=1, sticky="nsew")
        path = audio_files[self.audio_index]
        transcript = self.transcription_drafts.get(
            path, compilation.transcriptions.get(path, "")
        )
        if not transcript:
            sample, _ = mock_transcription(path)
            self.transcription_drafts[path] = sample
            transcript = sample

        audio_card = self._card(content)
        audio_card.pack(fill=X, pady=(0, 12))
        audio_header = Frame(audio_card, background=COLORS["surface"])
        audio_header.pack(fill=X)
        file_info = Frame(audio_header, background=COLORS["surface"])
        file_info.pack(side=LEFT, fill=X, expand=True)
        Label(
            file_info,
            text=display_filename(path, 34),
            font=("Segoe UI", 12, "bold"),
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
            font=("Segoe UI", 10),
            foreground=COLORS["muted"],
            background=COLORS["surface"],
        )
        self.audio_status_label.pack(anchor="w", pady=(4, 0))
        actions = Frame(audio_header, background=COLORS["surface"])
        actions.pack(side=RIGHT, padx=(8, 0))
        self._button(
            actions,
            "Save Edits",
            lambda selected=path: self._save_transcription(selected),
            success=True,
            compact=True,
        ).pack(side=RIGHT, padx=(6, 0))
        self._button(
            actions,
            "Reset sample",
            lambda selected=path: self._generate_transcription(selected),
            compact=True,
        ).pack(side=RIGHT, padx=(6, 0))
        self._button(
            actions,
            "Cancel",
            lambda selected=path: self._discard_transcription(selected),
            compact=True,
        ).pack(side=RIGHT)

        player = ctk.CTkFrame(
            audio_card,
            fg_color="#F8FAFC",
            corner_radius=10,
        )
        player.pack(fill=X, pady=(12, 0))
        player_controls = Frame(player, background="#F8FAFC", padx=12, pady=8)
        player_controls.pack(fill=X)
        self._button(
            player_controls,
            "Play",
            lambda selected=path: self._play_audio(selected),
            primary=True,
            compact=True,
        ).pack(side=LEFT)
        self._button(
            player_controls,
            "Pause",
            self._pause_audio,
            compact=True,
        ).pack(side=LEFT, padx=(8, 0))
        self._button(
            player_controls,
            "Stop",
            self._stop_audio,
            compact=True,
        ).pack(side=LEFT, padx=(8, 12))
        self.audio_time_label = Label(
            player_controls,
            text="0:00 / 0:00",
            font=("Segoe UI", 10),
            foreground=COLORS["muted"],
            background="#F8FAFC",
        )
        self.audio_time_label.pack(side=RIGHT)
        self.audio_slider = ctk.CTkSlider(
            player_controls,
            from_=0,
            to=1,
            number_of_steps=100,
            command=self._seek_audio,
            height=16,
            progress_color=COLORS["accent"],
            fg_color=COLORS["border"],
            button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_hover"],
        )
        self.audio_slider.configure(
            state="normal" if Path(path).suffix.lower() == ".mp3" else "disabled"
        )
        self.audio_slider.set(0)
        self.audio_slider.pack(side=LEFT, fill=X, expand=True)

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
        low_confidence = transcript.startswith(
            "[Low confidence]"
        ) or transcript.startswith("No speech could be detected")
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
        transcript_widget = TranscriptTextbox(
            transcript_card,
            wrap="word",
            font=("Segoe UI", 11),
            text_color=COLORS["body"],
            fg_color="#FAFBFC",
            border_color=COLORS["border"],
            border_width=1,
            corner_radius=8,
        )
        transcript_widget.pack(fill=BOTH, expand=True)
        transcript_widget.insert("1.0", transcript)
        transcript_widget._textbox.tag_configure(
            "timestamp", foreground=COLORS["muted"]
        )
        for timestamp in re.finditer(r"\[\d{1,2}:\d{2}\]", transcript):
            transcript_widget._textbox.tag_add(
                "timestamp",
                f"1.0 + {timestamp.start()} chars",
                f"1.0 + {timestamp.end()} chars",
            )
        self.transcription_editor = transcript_widget
        self.transcription_editor_path = path

    def _select_audio(self, index: int) -> None:
        self._stop_audio()
        compilation = self.active_compilation
        if (
            compilation is not None
            and self.transcription_editor is not None
            and self.transcription_editor.winfo_exists()
            and 0 <= self.audio_index < len(compilation.audio_files)
        ):
            current_path = compilation.audio_files[self.audio_index]
            self.transcription_drafts[current_path] = self.transcription_editor.get(
                "1.0", "end-1c"
            )
        self.audio_index = max(0, min(index, len(compilation.audio_files) - 1)) if compilation and compilation.audio_files else 0
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
        if (
            self.transcription_editor is not None
            and self.transcription_editor.winfo_exists()
        ):
            transcript = self.transcription_editor.get("1.0", "end-1c").strip()
        else:
            transcript = self.transcription_drafts.get(
                path, compilation.transcriptions.get(path, "")
            ).strip()
        if not transcript:
            messagebox.showinfo(
                "No transcript to save",
                "Enter or generate a transcript before saving it.",
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
        resolved_transcript = str(transcript_path.resolve())
        if resolved_transcript not in compilation.files:
            compilation.files.append(resolved_transcript)
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
            try:
                self.audio_duration_seconds = pygame.mixer.Sound(path).get_length()
            except pygame.error:
                self.audio_duration_seconds = 0.0
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            if self.audio_slider is not None:
                self.audio_slider.set(0)
            self._update_audio_position()
            if self.audio_status_label is not None:
                status = self.audio_status_label.cget("text")
                self.audio_status_label.configure(
                    text=status.replace("Ready to play", "Playing")
                )
            self._schedule_audio_position()
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
            if self.audio_position_after is not None:
                self.root.after_cancel(self.audio_position_after)
                self.audio_position_after = None
            if self.audio_status_label is not None:
                self.audio_status_label.configure(
                    text=self.audio_status_label.cget("text").replace(
                        "Playing", "Paused"
                    )
                )
        elif pygame.mixer.get_init() is not None:
            pygame.mixer.music.unpause()
            self._schedule_audio_position()
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
            if self.audio_position_after is not None:
                self.root.after_cancel(self.audio_position_after)
                self.audio_position_after = None
            if self.audio_slider is not None and self.audio_slider.winfo_exists():
                self.audio_slider.set(0)
            if self.audio_time_label is not None and self.audio_time_label.winfo_exists():
                self.audio_time_label.configure(
                    text=f"0:00 / {self._format_audio_time(self.audio_duration_seconds)}"
                )
            if self.audio_status_label is not None:
                self.audio_status_label.configure(
                    text=self.audio_status_label.cget("text")
                    .replace("Playing", "Ready to play")
                    .replace("Paused", "Ready to play")
                )

    @staticmethod
    def _format_audio_time(seconds: float) -> str:
        total_seconds = max(0, int(seconds))
        return f"{total_seconds // 60}:{total_seconds % 60:02d}"

    def _seek_audio(self, position: float) -> None:
        if self.audio_duration_seconds <= 0:
            return
        try:
            import pygame
        except ImportError:
            return
        if pygame.mixer.get_init() is None:
            return
        try:
            pygame.mixer.music.set_pos(self.audio_duration_seconds * position)
        except pygame.error as error:
            messagebox.showerror(
                "Unable to seek audio",
                f"Could not seek in the current audio file:\n{error}",
                parent=self.root,
            )

    def _schedule_audio_position(self) -> None:
        if self.audio_position_after is None:
            self.audio_position_after = self.root.after(500, self._update_audio_position)

    def _update_audio_position(self) -> None:
        self.audio_position_after = None
        try:
            import pygame
        except ImportError:
            return
        if pygame.mixer.get_init() is None:
            return
        position_ms = pygame.mixer.music.get_pos()
        if position_ms >= 0:
            position_seconds = position_ms / 1000
            if self.audio_slider is not None and self.audio_duration_seconds > 0:
                self.audio_slider.set(
                    min(1.0, position_seconds / self.audio_duration_seconds)
                )
            if self.audio_time_label is not None:
                self.audio_time_label.configure(
                    text=(
                        f"{self._format_audio_time(position_seconds)} / "
                        f"{self._format_audio_time(self.audio_duration_seconds)}"
                    )
                )
        if pygame.mixer.music.get_busy():
            self._schedule_audio_position()

    def _read_source_file_content(self, path: str | Path) -> str:
        target = Path(path)
        if not target.exists():
            return f"[Source file not found on disk:\n{target}]"
        suffix = target.suffix.lower()
        if suffix == ".txt":
            try:
                return target.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    return target.read_text(encoding="latin-1", errors="replace")
                except Exception as error:
                    return f"[Unable to read text file:\n{error}]"
        elif suffix == ".pdf":
            return (
                f"[PDF Document: {target.name}]\n"
                f"Path: {target}\n"
                f"Size: {target.stat().st_size:,} bytes\n\n"
                "This is a PDF document. To view the original formatted layout and pages, "
                "click 'Open file' above to view in your default PDF reader."
            )
        return f"[File type {suffix} is not a document file]"

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

        # Body packed fill=BOTH, expand=True
        body = Frame(self.page_container, background=COLORS["background"])
        body.pack(side="top", fill=BOTH, expand=True, padx=24, pady=(0, 10))
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # Left files panel (sidebar file traversal buttons)
        files_panel = self._card(body, width=250)
        files_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        files_panel.grid_propagate(False)
        Label(
            files_panel,
            text="Compilation files",
            font=("Segoe UI", 11, "bold"),
            foreground=COLORS["text"],
            background=COLORS["surface"],
        ).pack(anchor="w", pady=(0, 10))
        for index, s_path in enumerate(files):
            done = s_path in self.summary_drafts and bool(self.summary_drafts[s_path].strip())
            self._button(
                files_panel,
                f"{'Ready' if done else 'Pending'}  ·  {display_filename(s_path, 28)}",
                lambda selected=index: self._select_summary_file(selected),
                primary=index == self.summary_index,
                pady=11,
                font_size=10,
            ).pack(fill=X, pady=(0, 7))

        content = Frame(body, background=COLORS["background"])
        content.grid(row=0, column=1, sticky="nsew")
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=1)
        path = files[self.summary_index]
        summary_card = self._card(content)
        summary_card.grid(row=0, column=0, sticky="nsew")

        card_header = Frame(summary_card, background=COLORS["surface"])
        card_header.pack(fill=X, pady=(0, 10))
        Label(
            card_header,
            text=display_filename(path, 42),
            font=("Segoe UI", 13, "bold"),
            foreground=COLORS["text"],
            background=COLORS["surface"],
        ).pack(side=LEFT)
        summary_text = self.summary_drafts.get(path, "")
        if not summary_text:
            transcription = compilation.transcriptions.get(path, "")
            summary_text = mock_summary(path, transcription)
            self.summary_drafts[path] = summary_text

        split_area = Frame(summary_card, background=COLORS["surface"])
        split_area.pack(fill=BOTH, expand=True)
        split_area.columnconfigure(0, weight=1, uniform="split")
        split_area.columnconfigure(1, weight=1, uniform="split")
        split_area.rowconfigure(0, weight=1)

        # Left pane: Source File
        left_pane = ctk.CTkFrame(
            split_area,
            fg_color=COLORS["surface"],
            border_color=COLORS["border"],
            border_width=1,
            corner_radius=8,
        )
        left_pane.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        left_header = Frame(left_pane, background="#F1F5F9")
        left_header.pack(fill=X, pady=(0, 6))
        Label(
            left_header,
            text="Source Text File",
            font=("Segoe UI", 10, "bold"),
            foreground=COLORS["text"],
            background="#F1F5F9",
        ).pack(side=LEFT)
        self._button(
            left_header,
            "Open file",
            lambda selected=path: self._open_file(selected),
            primary=True,
            compact=True,
        ).pack(side=RIGHT)

        source_text_area = Frame(left_pane, background=COLORS["surface"])
        source_text_area.pack(fill=BOTH, expand=True)
        source_widget = Text(
            source_text_area,
            wrap="word",
            font=("Segoe UI", 10),
            foreground=COLORS["text"],
            background="#F8F9FA",
            relief="flat",
            padx=10,
            pady=8,
        )
        source_scroll = AutoHideScrollbar(
            source_text_area, orient=VERTICAL, command=source_widget.yview
        )
        source_widget.configure(yscrollcommand=source_scroll.set)
        source_widget.pack(side=LEFT, fill=BOTH, expand=True)
        source_scroll.pack(side=RIGHT, fill=Y)
        source_widget.insert("1.0", self._read_source_file_content(path))
        source_widget.configure(state="disabled")

        # Right pane: Generated Summary
        right_pane = ctk.CTkFrame(
            split_area,
            fg_color=COLORS["surface"],
            border_color=COLORS["border"],
            border_width=1,
            corner_radius=8,
        )
        right_pane.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        right_header = Frame(right_pane, background="#EEF2FF")
        right_header.pack(fill=X, pady=(0, 6))
        Label(
            right_header,
            text="Generated Summary",
            font=("Segoe UI", 10, "bold"),
            foreground=COLORS["text"],
            background="#EEF2FF",
        ).pack(side=LEFT)
        self._button(
            right_header,
            "Regenerate",
            lambda selected=path: self._generate_summary(selected),
            primary=True,
            compact=True,
        ).pack(side=RIGHT)

        summary_area = Frame(right_pane, background=COLORS["surface"])
        summary_area.pack(fill=BOTH, expand=True)
        summary_widget = Text(
            summary_area,
            wrap="word",
            font=("Segoe UI", 10),
            foreground=COLORS["text"],
            background="#FAFBFC",
            relief="solid",
            borderwidth=1,
            padx=10,
            pady=8,
        )
        summary_scroll = AutoHideScrollbar(
            summary_area, orient=VERTICAL, command=summary_widget.yview
        )
        summary_widget.configure(yscrollcommand=summary_scroll.set)
        summary_widget.pack(side=LEFT, fill=BOTH, expand=True)
        summary_scroll.pack(side=RIGHT, fill=Y)
        summary_widget.insert("1.0", summary_text)
        self.summary_editor = summary_widget
        self.summary_editor_path = path
        self.summary_progress = ctk.CTkProgressBar(
            right_pane,
            mode="indeterminate",
            height=5,
            progress_color=COLORS["accent"],
            fg_color=COLORS["accent_light"],
        )

        # Controls right below the split screen:
        controls = Frame(content, background=COLORS["background"])
        controls.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        left_controls = Frame(controls, background=COLORS["background"])
        left_controls.pack(side=LEFT)
        right_controls = Frame(controls, background=COLORS["background"])
        right_controls.pack(side=RIGHT)

        has_next = self.summary_index < len(files) - 1
        if has_next:
            self._button(
                right_controls,
                "Next File",
                self._next_summary_file,
                primary=True,
            ).pack(side=RIGHT)
            self._button(
                left_controls,
                "Save",
                self._save_summary_edits,
            ).pack(side=LEFT, padx=(0, 8))
        else:
            self._button(
                right_controls,
                "Finalize",
                self._finalize_summary,
                primary=True,
            ).pack(side=RIGHT)

        self._button(
            left_controls,
            "Cancel",
            self._cancel_summary_edits,
        ).pack(side=LEFT)

        if self.summary_index > 0:
            self._button(
                right_controls,
                "Previous",
                self._previous_summary_file,
            ).pack(side=RIGHT, padx=(0, 8))

    def _next_summary_file(self) -> None:
        compilation = self._require_compilation()
        if compilation is None:
            return
        self._stage_current_summary()
        files = compilation.summary_files
        if self.summary_index < len(files) - 1:
            self._select_summary_file(self.summary_index + 1)

    def _previous_summary_file(self) -> None:
        compilation = self._require_compilation()
        if compilation is None:
            return
        self._stage_current_summary()
        if self.summary_index > 0:
            self._select_summary_file(self.summary_index - 1)

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
            or self.summary_streaming_path == path
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
        compilation = self._require_compilation()
        if compilation is None:
            return
        self._stage_current_summary()
        generated = mock_summary(
            path, compilation.transcriptions.get(path, "")
        )
        editor = self.summary_editor
        progress = self.summary_progress
        if (
            editor is None
            or progress is None
            or not editor.winfo_exists()
            or self.summary_editor_path != path
        ):
            self.summary_drafts[path] = generated
            self.show_page("summary")
            return
        editor.configure(state="normal")
        editor.delete("1.0", "end")
        self.summary_drafts[path] = generated
        self.summary_streaming_path = path
        self.summary_stream_generation += 1
        generation = self.summary_stream_generation
        progress.pack(fill=X, pady=(8, 0))
        progress.start()
        self._stream_summary_text(path, generated, generation)

    def _stream_summary_text(
        self, path: str, generated: str, generation: int, offset: int = 0
    ) -> None:
        if generation != self.summary_stream_generation:
            return
        editor = self.summary_editor
        progress = self.summary_progress
        if (
            editor is None
            or progress is None
            or not editor.winfo_exists()
            or self.summary_editor_path != path
        ):
            return
        next_offset = min(offset + 48, len(generated))
        editor.insert("end", generated[offset:next_offset])
        editor.see("end")
        if next_offset < len(generated):
            self.summary_stream_after = self.root.after(
                24,
                lambda: self._stream_summary_text(
                    path, generated, generation, next_offset
                ),
            )
            return
        self.summary_stream_after = None
        self.summary_streaming_path = None
        progress.stop()
        progress.pack_forget()

    def _finalize_summary(self) -> None:
        compilation = self._require_compilation()
        if compilation is None:
            return
        self._stage_current_summary()
        for path in compilation.summary_files:
            if not self.summary_drafts.get(path, "").strip():
                transcription = compilation.transcriptions.get(path, "")
                self.summary_drafts[path] = mock_summary(path, transcription)
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
        scrollbar = AutoHideScrollbar(
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
        card.pack(fill=BOTH, expand=True, padx=24, pady=(0, 18))
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
            message_background = COLORS["accent"] if speaker == "You" else "#F1F5F9"
            message = ctk.CTkFrame(
                conversation.content,
                fg_color=message_background,
                corner_radius=10,
            )
            message.pack(
                anchor="e" if speaker == "You" else "w",
                padx=(48, 4) if speaker == "You" else (4, 48),
                pady=(0, 10),
            )
            Label(
                message,
                text=speaker,
                font=("Segoe UI", 10, "bold"),
                foreground="#FFFFFF" if speaker == "You" else COLORS["muted"],
                background=message_background,
                padx=12,
                pady=4,
            ).pack(anchor="w")
            Label(
                message,
                text=text,
                font=("Segoe UI", 11),
                foreground="#FFFFFF" if speaker == "You" else COLORS["body"],
                background=message_background,
                justify=LEFT,
                wraplength=560,
                padx=12,
                pady=8,
            ).pack(anchor="w")
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
        entry = ctk.CTkTextbox(
            composer,
            font=("Segoe UI", 11),
            wrap="word",
            height=64,
            corner_radius=9,
            border_width=1,
            border_color=COLORS["border"],
            fg_color="#FAFBFC",
            text_color=COLORS["text"],
        )
        self.chat_entry = entry
        entry.pack(side=LEFT, fill=X, expand=True, padx=(0, 10))

        self._button(
            composer, "Send", self._send_chat_message, primary=True
        ).pack(side=RIGHT)
        self._button(
            composer, "Close", self._close_chat
        ).pack(side=RIGHT, padx=(0, 8))
        ctk.CTkLabel(
            card,
            text="Enter to send  ·  Shift + Enter for a new line",
            font=("Segoe UI", 9),
            text_color=COLORS["muted"],
            anchor="w",
        ).pack(anchor="w", pady=(6, 0))

        def handle_chat_return(event: object) -> str | None:
            if getattr(event, "state", 0) & 0x0001:
                return None
            self._send_chat_message()
            return "break"

        entry.bind("<Return>", handle_chat_return)
        entry.focus_set()

    def _send_chat_message(self) -> None:
        question = (
            self.chat_entry.get("1.0", "end-1c").strip()
            if self.chat_entry
            else ""
        )
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
