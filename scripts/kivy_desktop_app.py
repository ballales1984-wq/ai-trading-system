#!/usr/bin/env python3
"""
Kivy desktop shell for AI Trading System.

MVP features:
- Start/stop local FastAPI backend
- Open dashboard in browser
- Save API keys locally on user machine
"""

from __future__ import annotations

import ctypes
import json
import os
import socket
import sys
import threading
import traceback
import webbrowser
from datetime import datetime
from ctypes import wintypes
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

os.environ.setdefault("KIVY_NO_FILELOG", "1")

import uvicorn
from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout


_SINGLE_INSTANCE_MUTEX_NAME = "Local\\AITradingSystemDesktopMutex"
_ERROR_ALREADY_EXISTS = 183
_app_mutex_handle = None

try:
    from tkinter import Tk, filedialog
except Exception:
    Tk = None
    filedialog = None


KV = """
<RootWidget>:
    orientation: "vertical"
    padding: 16
    spacing: 10

    Label:
        text: "AI Trading System - Desktop (Kivy)"
        font_size: "22sp"
        size_hint_y: None
        height: "40dp"

    Label:
        text: root.instructions_text
        size_hint_y: None
        height: "130dp"
        text_size: self.width, None
        halign: "left"
        valign: "top"

    Label:
        id: status_label
        text: root.status_text
        color: (0.2, 0.8, 0.2, 1) if root.backend_running else (1, 0.5, 0.3, 1)
        size_hint_y: None
        height: "30dp"

    BoxLayout:
        size_hint_y: None
        height: "42dp"
        spacing: 8
        Button:
            text: "Start Backend"
            on_release: root.start_backend()
        Button:
            text: "Stop Backend"
            on_release: root.stop_backend()
        Button:
            text: "Open Dashboard"
            on_release: root.open_dashboard()

    Label:
        text: "Local API Keys (saved only on this PC)"
        size_hint_y: None
        height: "28dp"

    GridLayout:
        cols: 2
        size_hint_y: None
        height: "130dp"
        row_force_default: True
        row_default_height: "38dp"
        spacing: 8

        Label:
            text: "BINANCE_API_KEY"
            halign: "left"
            text_size: self.size
        TextInput:
            id: binance_key
            multiline: False
            password: True

        Label:
            text: "BINANCE_SECRET_KEY"
            halign: "left"
            text_size: self.size
        TextInput:
            id: binance_secret
            multiline: False
            password: True

        Label:
            text: "NEWSAPI_KEY"
            halign: "left"
            text_size: self.size
        TextInput:
            id: newsapi_key
            multiline: False
            password: True

    BoxLayout:
        size_hint_y: None
        height: "42dp"
        spacing: 8
        Button:
            text: "Save Keys"
            on_release: root.save_keys()
        Button:
            text: "Load Keys"
            on_release: root.load_keys()
        Button:
            text: "Load .env"
            on_release: root.load_env_file()
        Button:
            text: "Set .env Folder"
            on_release: root.set_env_folder()
        Button:
            text: "Create .env Template"
            on_release: root.create_env_template()

    Label:
        id: message_label
        text: root.message_text
        size_hint_y: None
        height: "56dp"
        text_size: self.width, None
        halign: "left"
        valign: "middle"

    Label:
        text: root.disclaimer_text
        size_hint_y: None
        height: "46dp"
        text_size: self.width, None
        halign: "left"
        valign: "middle"
        color: 1, 0.65, 0.2, 1
"""


def _is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


class DATA_BLOB(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


_crypt32 = ctypes.windll.crypt32
_kernel32 = ctypes.windll.kernel32
_CRYPTPROTECT_UI_FORBIDDEN = 0x01


def _bytes_from_blob(blob: DATA_BLOB) -> bytes:
    if blob.cbData == 0 or not blob.pbData:
        return b""
    return ctypes.string_at(blob.pbData, blob.cbData)


def _dpapi_encrypt(raw: bytes) -> bytes:
    in_buf = (ctypes.c_byte * len(raw)).from_buffer_copy(raw) if raw else None
    in_blob = DATA_BLOB(
        len(raw),
        ctypes.cast(in_buf, ctypes.POINTER(ctypes.c_byte)) if in_buf is not None else None,
    )
    out_blob = DATA_BLOB()
    ok = _crypt32.CryptProtectData(
        ctypes.byref(in_blob),
        "AITradingSystemKeys",
        None,
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(out_blob),
    )
    if not ok:
        raise OSError(f"CryptProtectData failed: {ctypes.GetLastError()}")
    try:
        return _bytes_from_blob(out_blob)
    finally:
        _kernel32.LocalFree(ctypes.cast(out_blob.pbData, ctypes.c_void_p))


def _dpapi_decrypt(raw: bytes) -> bytes:
    in_buf = (ctypes.c_byte * len(raw)).from_buffer_copy(raw) if raw else None
    in_blob = DATA_BLOB(
        len(raw),
        ctypes.cast(in_buf, ctypes.POINTER(ctypes.c_byte)) if in_buf is not None else None,
    )
    out_blob = DATA_BLOB()
    ok = _crypt32.CryptUnprotectData(
        ctypes.byref(in_blob),
        None,
        None,
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(out_blob),
    )
    if not ok:
        raise OSError(f"CryptUnprotectData failed: {ctypes.GetLastError()}")
    try:
        return _bytes_from_blob(out_blob)
    finally:
        _kernel32.LocalFree(ctypes.cast(out_blob.pbData, ctypes.c_void_p))


class RootWidget(BoxLayout):
    host = "127.0.0.1"
    port = 8000
    backend_running = False
    status_text = "Backend status: stopped"
    message_text = "Tip: use 'Create .env Template' then 'Load .env'."
    instructions_text = (
        "Quick start for client API keys:\\n"
        "1) Click 'Create .env Template'.\\n"
        "2) Open the generated .env and fill your keys.\\n"
        "3) Click 'Set .env Folder' once (or 'Load .env').\\n"
        "4) Keys auto-load at startup and are saved encrypted locally.\\n"
        "5) Click 'Start Backend' then 'Open Dashboard'."
    )
    disclaimer_text = (
        "Disclaimer: Educational/informational software only. "
        "Not investment advice. Trading involves risk of capital loss."
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._server = None
        self._server_thread = None
        self._backend_start_wait_ticks = 0
        self._backend_start_applied = 0
        _append_runtime_log("RootWidget initialized")
        Clock.schedule_once(lambda *_: self.bootstrap_keys(), 0.2)

    @property
    def config_path(self) -> Path:
        base = Path.home() / "AppData" / "Local" / "AITradingSystem"
        base.mkdir(parents=True, exist_ok=True)
        return base / "desktop_settings.enc"

    @property
    def legacy_config_path(self) -> Path:
        base = Path.home() / "AppData" / "Local" / "AITradingSystem"
        return base / "desktop_settings.json"

    @property
    def state_path(self) -> Path:
        base = Path.home() / "AppData" / "Local" / "AITradingSystem"
        base.mkdir(parents=True, exist_ok=True)
        return base / "desktop_state.json"

    def _set_message(self, text: str) -> None:
        self.message_text = text
        self.ids.message_label.text = text

    def _set_status(self, running: bool) -> None:
        self.backend_running = running
        self.status_text = (
            f"Backend status: running on http://{self.host}:{self.port}"
            if running
            else "Backend status: stopped"
        )
        self.ids.status_label.text = self.status_text
        self.ids.status_label.color = (0.2, 0.8, 0.2, 1) if running else (1, 0.5, 0.3, 1)

    def _run_server(self) -> None:
        config = uvicorn.Config(
            "app.main:app",
            host=self.host,
            port=self.port,
            reload=False,
            log_level="info",
        )
        self._server = uvicorn.Server(config)
        self._server.run()

    def _collect_keys_from_ui(self) -> Dict[str, str]:
        return {
            "BINANCE_API_KEY": self.ids.binance_key.text.strip(),
            "BINANCE_SECRET_KEY": self.ids.binance_secret.text.strip(),
            "NEWSAPI_KEY": self.ids.newsapi_key.text.strip(),
        }

    def _normalize_payload_keys(self, payload: Dict[str, str]) -> Dict[str, str]:
        aliases = {
            "BINANCE_API_KEY": "BINANCE_API_KEY",
            "BINANCE_KEY": "BINANCE_API_KEY",
            "BINANCE_SECRET_KEY": "BINANCE_SECRET_KEY",
            "BINANCE_API_SECRET": "BINANCE_SECRET_KEY",
            "BINANCE_SECRET": "BINANCE_SECRET_KEY",
            "NEWSAPI_KEY": "NEWSAPI_KEY",
            "NEWS_API_KEY": "NEWSAPI_KEY",
        }
        normalized: Dict[str, str] = {}
        for raw_key, raw_value in payload.items():
            key = str(raw_key).strip().upper()
            if not key:
                continue
            canonical = aliases.get(key)
            if canonical is None:
                continue
            value = str(raw_value).strip()
            if value:
                normalized[canonical] = value
        return normalized

    def _apply_payload_to_ui(self, payload: Dict[str, str]) -> int:
        normalized = self._normalize_payload_keys(payload)
        self.ids.binance_key.text = normalized.get("BINANCE_API_KEY", self.ids.binance_key.text)
        self.ids.binance_secret.text = normalized.get(
            "BINANCE_SECRET_KEY", self.ids.binance_secret.text
        )
        self.ids.newsapi_key.text = normalized.get("NEWSAPI_KEY", self.ids.newsapi_key.text)
        return sum(
            bool(normalized.get(k, "").strip())
            for k in ("BINANCE_API_KEY", "BINANCE_SECRET_KEY", "NEWSAPI_KEY")
        )

    def _apply_keys_to_env(self) -> int:
        keys = self._collect_keys_from_ui()
        applied = 0
        for k, v in keys.items():
            if v:
                os.environ[k] = v
                applied += 1
        return applied

    def _parse_env_text(self, raw_text: str) -> Dict[str, str]:
        payload: Dict[str, str] = {}
        for raw_line in raw_text.splitlines():
            line = raw_line.strip().lstrip("\ufeff")
            if line.lower().startswith("export "):
                line = line[7:].strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            payload[key] = value
        return payload

    def _load_state(self) -> Dict[str, str]:
        if not self.state_path.exists():
            return {}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_state(self, updates: Dict[str, str]) -> None:
        state = self._load_state()
        state.update({k: v for k, v in updates.items() if v})
        self.state_path.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")

    def _import_env_path(
        self,
        env_path: Path,
        *,
        announce: bool = True,
        auto_save_encrypted: bool = True,
    ) -> int:
        try:
            raw = env_path.read_text(encoding="utf-8")
            payload = self._normalize_payload_keys(self._parse_env_text(raw))
        except Exception as exc:
            if announce:
                self._set_message(f"Cannot read .env file: {exc}")
            return 0

        imported = self._apply_payload_to_ui(payload)
        if imported == 0:
            if announce:
                self._set_message(
                    "Loaded file but no supported keys found. Expected: "
                    "BINANCE_API_KEY, BINANCE_SECRET_KEY, NEWSAPI_KEY."
                )
            return 0

        missing = [
            k
            for k in ("BINANCE_API_KEY", "BINANCE_SECRET_KEY", "NEWSAPI_KEY")
            if not payload.get(k, "").strip()
        ]
        self._save_state({"env_file": str(env_path), "env_folder": str(env_path.parent)})

        if auto_save_encrypted:
            self.save_keys(silent=True)

        if announce:
            if missing:
                self._set_message(
                    f"Loaded .env from {env_path} ({imported} keys). "
                    f"Missing: {', '.join(missing)}."
                )
            else:
                self._set_message(
                    f"Loaded .env from {env_path} ({imported} keys) and saved encrypted."
                )
        return imported

    def _try_os_env(self) -> int:
        payload = {
            "BINANCE_API_KEY": os.getenv("BINANCE_API_KEY", ""),
            "BINANCE_SECRET_KEY": os.getenv("BINANCE_SECRET_KEY", ""),
            "NEWSAPI_KEY": os.getenv("NEWSAPI_KEY", ""),
        }
        imported = self._apply_payload_to_ui(payload)
        if imported > 0:
            self.save_keys(silent=True)
        return imported

    def _auto_env_candidates(self) -> list[Path]:
        state = self._load_state()
        candidates: list[Path] = []
        env_file = state.get("env_file", "").strip()
        env_folder = state.get("env_folder", "").strip()
        if env_file:
            candidates.append(Path(env_file))
        if env_folder:
            candidates.append(Path(env_folder) / ".env")

        if getattr(sys, "frozen", False):
            exe_dir = Path(sys.executable).resolve().parent
            candidates.append(exe_dir / ".env")

        candidates.append(Path.cwd() / ".env")

        dedup: list[Path] = []
        seen: set[str] = set()
        for p in candidates:
            key = str(p).lower()
            if key not in seen:
                seen.add(key)
                dedup.append(p)
        return dedup

    def _auto_import_env(self) -> Tuple[int, Optional[Path]]:
        for env_path in self._auto_env_candidates():
            if env_path.exists():
                imported = self._import_env_path(
                    env_path, announce=False, auto_save_encrypted=True
                )
                if imported > 0:
                    return imported, env_path
        return 0, None

    def bootstrap_keys(self) -> None:
        loaded = self.load_keys(silent=True, fallback_to_env=False)
        if loaded > 0:
            self._set_message(f"Keys loaded from local encrypted storage ({loaded} keys).")
            return

        from_env = self._try_os_env()
        if from_env > 0:
            self._set_message(
                f"Loaded {from_env} keys from OS environment and saved encrypted."
            )
            return

        from_file, env_path = self._auto_import_env()
        if from_file > 0 and env_path is not None:
            self._set_message(
                f"Auto-loaded {from_file} keys from {env_path} and saved encrypted."
            )
            return

        self._set_message(
            "No local keys file yet. Use 'Set .env Folder' once or click 'Load .env'."
        )

    def create_env_template(self) -> None:
        template = (
            "# AI Trading System - Client .env template\\n"
            "# Fill values and save file, then load it with 'Load .env'\\n\\n"
            "BINANCE_API_KEY=\\n"
            "BINANCE_SECRET_KEY=\\n"
            "NEWSAPI_KEY=\\n"
        )
        target = self.config_path.parent / "client_template.env"
        target.write_text(template, encoding="utf-8")
        self._set_message(f".env template created: {target}")

    def start_backend(self) -> None:
        if self._server_thread and self._server_thread.is_alive():
            self._set_message("Backend already running.")
            return

        self._backend_start_applied = self._apply_keys_to_env()
        self._backend_start_wait_ticks = 0
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()
        self._set_message("Starting backend...")
        Clock.schedule_interval(self._poll_backend_start, 0.1)

    def _poll_backend_start(self, _dt: float) -> bool:
        self._backend_start_wait_ticks += 1
        if _is_port_open(self.host, self.port):
            self._set_status(True)
            if self._backend_start_applied == 0:
                self._set_message(
                    "Backend started with 0 keys. Load .env or type keys before live data."
                )
            else:
                self._set_message(
                    f"Backend started. Loaded {self._backend_start_applied} local API keys."
                )
            return False

        if self._backend_start_wait_ticks >= 100:
            self._set_status(False)
            self._set_message(
                "Backend did not start in time. Check desktop_crash.log and verify port 8000 is free."
            )
            return False

        return True

    def stop_backend(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=5)
        self._set_status(False)
        self._set_message("Backend stopped.")

    def open_dashboard(self) -> None:
        if not _is_port_open(self.host, self.port):
            self._set_message("Backend not running. Start backend first.")
            return
        webbrowser.open(f"http://{self.host}:{self.port}/dashboard")
        self._set_message("Dashboard opened in browser.")

    def save_keys(self, silent: bool = False) -> None:
        payload = self._collect_keys_from_ui()
        raw = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        encrypted = _dpapi_encrypt(raw)
        self.config_path.write_bytes(encrypted)
        if not silent:
            self._set_message(f"Keys saved encrypted in {self.config_path}")

    def load_keys(self, silent: bool = False, fallback_to_env: bool = True) -> int:
        payload: Optional[Dict[str, str]] = None
        decrypt_error: Optional[str] = None

        if self.config_path.exists():
            try:
                encrypted = self.config_path.read_bytes()
                raw = _dpapi_decrypt(encrypted)
                payload = json.loads(raw.decode("utf-8"))
            except (ValueError, OSError) as exc:
                decrypt_error = str(exc)

        if payload is None and self.legacy_config_path.exists():
            try:
                payload = json.loads(self.legacy_config_path.read_text(encoding="utf-8"))
                # Migrate once to encrypted storage.
                self.config_path.write_bytes(_dpapi_encrypt(json.dumps(payload).encode("utf-8")))
                self.legacy_config_path.unlink(missing_ok=True)
            except (ValueError, OSError) as exc:
                if decrypt_error is None:
                    decrypt_error = str(exc)

        if payload is not None:
            loaded = self._apply_payload_to_ui(payload)
            if not silent:
                self._set_message(f"Keys loaded from local encrypted storage ({loaded} keys).")
            return loaded

        if fallback_to_env:
            from_file, env_path = self._auto_import_env()
            if from_file > 0 and env_path is not None:
                if not silent:
                    self._set_message(f"Keys loaded from .env file ({from_file} keys): {env_path}")
                return from_file

            from_os = self._try_os_env()
            if from_os > 0:
                if not silent:
                    self._set_message(
                        f"Keys loaded from OS environment ({from_os} keys) and saved encrypted."
                    )
                return from_os

        if not silent:
            if decrypt_error:
                self._set_message(
                    "Cannot read encrypted keys for this user/machine. "
                    "Use 'Load .env' or 'Set .env Folder'."
                )
            else:
                self._set_message("No local keys file yet. Use 'Load .env' or 'Set .env Folder'.")
        return 0

    def set_env_folder(self) -> None:
        if Tk is None or filedialog is None:
            self._set_message("File dialog unavailable. Install tkinter support.")
            return
        try:
            root = Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            folder = filedialog.askdirectory(title="Select folder containing .env")
            root.destroy()
        except Exception as exc:
            self._set_message(f"Cannot open folder chooser: {exc}")
            return

        if not folder:
            self._set_message("No folder selected.")
            return

        env_path = Path(folder) / ".env"
        self._save_state({"env_folder": folder})
        if not env_path.exists():
            self._set_message(f"Folder saved. Missing file: {env_path}")
            return

        self._import_env_path(env_path, announce=True, auto_save_encrypted=True)

    def load_env_file(self) -> None:
        if Tk is None or filedialog is None:
            self._set_message("File dialog unavailable. Install tkinter support.")
            return
        try:
            root = Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            env_path = filedialog.askopenfilename(
                title="Select .env file",
                filetypes=[("Env files", "*.env"), ("All files", "*.*")],
            )
            root.destroy()
        except Exception as exc:
            self._set_message(f"Cannot open file chooser: {exc}")
            return

        if not env_path:
            self._set_message("No .env file selected.")
            return

        self._import_env_path(Path(env_path), announce=True, auto_save_encrypted=True)


class KivyDesktopApp(App):
    title = "AI Trading System Desktop"

    def build(self) -> RootWidget:
        _append_runtime_log("Kivy build start")
        Builder.load_string(KV)
        root = RootWidget()
        Clock.schedule_once(lambda *_: self._bring_to_front(), 0.3)
        _append_runtime_log("Kivy build complete")
        return root

    def on_start(self) -> None:
        _append_runtime_log("Kivy app started")
        self._bring_to_front()

    def _bring_to_front(self) -> None:
        try:
            from kivy.core.window import Window

            Window.raise_window()
        except Exception as exc:
            _append_runtime_log(f"Window raise failed: {exc}")

    def on_stop(self) -> None:
        _append_runtime_log("Kivy app stopped")
        root = self.root
        if root:
            root.stop_backend()


def _runtime_dir() -> Path:
    base = Path.home() / "AppData" / "Local" / "AITradingSystem"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _append_runtime_log(message: str) -> None:
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with (_runtime_dir() / "desktop_runtime.log").open("a", encoding="utf-8") as fh:
            fh.write(f"[{ts}] {message}\n")
    except Exception:
        pass


def _write_crash_log(exc_text: str) -> Path:
    log_path = _runtime_dir() / "desktop_crash.log"
    log_path.write_text(exc_text, encoding="utf-8")
    return log_path


def _show_error_dialog(message: str) -> None:
    try:
        ctypes.windll.user32.MessageBoxW(None, message, "AI Trading System Desktop Error", 0x10)
    except Exception:
        pass


def _acquire_single_instance_lock() -> bool:
    global _app_mutex_handle
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.GetLastError.restype = wintypes.DWORD
        handle = kernel32.CreateMutexW(None, False, _SINGLE_INSTANCE_MUTEX_NAME)
        if not handle:
            return True
        _app_mutex_handle = handle
        return kernel32.GetLastError() != _ERROR_ALREADY_EXISTS
    except Exception:
        return True


if __name__ == "__main__":
    try:
        if not _acquire_single_instance_lock():
            _show_error_dialog("AI Trading System Desktop is already running.")
            raise SystemExit(0)
        _append_runtime_log("Desktop app bootstrap")
        KivyDesktopApp().run()
    except Exception:
        error_text = traceback.format_exc()
        path = _write_crash_log(error_text)
        _show_error_dialog(f"Application crash. Log saved to:\n{path}")
        raise
