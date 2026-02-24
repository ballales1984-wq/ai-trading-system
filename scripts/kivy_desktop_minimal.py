#!/usr/bin/env python3
"""
Minimal Kivy desktop shell for AI Trading System.
This is a stripped-down version for faster PyInstaller building.
The actual API backend will run as a subprocess.
"""

from __future__ import annotations

import ctypes
import json
import os
import socket
import sys
import threading
import time
import traceback
import webbrowser
from ctypes import wintypes
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Only import Kivy at runtime, not at import time
KV = """
<RootWidget>:
    orientation: "vertical"
    padding: 16
    spacing: 10

    Label:
        text: "AI Trading System - Desktop"
        font_size: "22sp"
        size_hint_y: None
        height: "40dp"

    Label:
        text: root.instructions_text
        size_hint_y: None
        height: "100dp"
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
        text: "API Keys (saved locally encrypted)"
        size_hint_y: None
        height: "28dp"

    GridLayout:
        cols: 2
        size_hint_y: None
        height: "80dp"
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

    Label:
        id: message_label
        text: root.message_text
        size_hint_y: None
        height: "40dp"
        text_size: self.width, None
        halign: "left"
        valign: "middle"
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


class RootWidget:
    """Non-Kivy base class for type hints"""
    host = "127.0.0.1"
    port = 8000
    backend_running = False
    status_text = "Backend status: stopped"
    message_text = "Tip: Load .env or enter keys and Save."
    instructions_text = "Quick start:\\n1) Enter API keys\\n2) Click Save Keys\\n3) Start Backend\\n4) Open Dashboard"
    ids = {}


# Lazy import to speed up PyInstaller analysis
_kivy_app_class = None


def _get_kivy_app():
    global _kivy_app_class
    if _kivy_app_class is None:
        from kivy.app import App
        from kivy.clock import Clock
        from kivy.lang import Builder
        from kivy.uix.boxlayout import BoxLayout
        
        class RootWidgetImpl(BoxLayout):
            host = "127.0.0.1"
            port = 8000
            backend_running = False
            status_text = "Backend status: stopped"
            message_text = "Tip: Load .env or enter keys and Save."
            instructions_text = "Quick start:\\n1) Enter API keys\\n2) Click Save Keys\\n3) Start Backend\\n4) Open Dashboard"

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._server_process = None
                Clock.schedule_once(lambda *_: self.bootstrap_keys(), 0.2)

            @property
            def config_path(self):
                base = Path.home() / "AppData" / "Local" / "AITradingSystem"
                base.mkdir(parents=True, exist_ok=True)
                return base / "desktop_settings.enc"

            def _set_message(self, text):
                self.message_text = text

            def _set_status(self, running):
                self.backend_running = running
                self.status_text = (
                    f"Backend running on http://{self.host}:{self.port}"
                    if running
                    else "Backend status: stopped"
                )

            def _collect_keys_from_ui(self):
                return {
                    "BINANCE_API_KEY": self.ids.binance_key.text.strip(),
                    "BINANCE_SECRET_KEY": self.ids.binance_secret.text.strip(),
                }

            def _apply_payload_to_ui(self, payload):
                self.ids.binance_key.text = payload.get("BINANCE_API_KEY", "")
                self.ids.binance_secret.text = payload.get("BINANCE_SECRET_KEY", "")

            def _run_server_subprocess(self):
                """Run backend as subprocess to avoid import issues"""
                import subprocess
                env = os.environ.copy()
                keys = self._collect_keys_from_ui()
                for k, v in keys.items():
                    if v:
                        env[k] = v
                
                # Start the backend
                python_exe = sys.executable
                self._server_process = subprocess.Popen(
                    [python_exe, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
                    env=env,
                    cwd=str(Path.cwd()),
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                )

            def start_backend(self):
                if self.backend_running:
                    self._set_message("Backend already running.")
                    return

                self._run_server_subprocess()
                
                # Wait for port to be ready
                for _ in range(50):
                    if _is_port_open(self.host, self.port):
                        self._set_status(True)
                        self._set_message("Backend started.")
                        return
                    time.sleep(0.2)

                self._set_status(False)
                self._set_message("Backend failed to start.")

            def stop_backend(self):
                if self._server_process:
                    self._server_process.terminate()
                    self._server_process = None
                self._set_status(False)
                self._set_message("Backend stopped.")

            def open_dashboard(self):
                if not _is_port_open(self.host, self.port):
                    self._set_message("Backend not running.")
                    return
                webbrowser.open(f"http://{self.host}:{self.port}/dashboard")

            def save_keys(self):
                payload = self._collect_keys_from_ui()
                raw = json.dumps(payload, ensure_ascii=True).encode("utf-8")
                encrypted = _dpapi_encrypt(raw)
                self.config_path.write_bytes(encrypted)
                self._set_message(f"Keys saved encrypted.")

            def load_keys(self):
                if not self.config_path.exists():
                    self._set_message("No saved keys.")
                    return
                try:
                    encrypted = self.config_path.read_bytes()
                    raw = _dpapi_decrypt(encrypted)
                    payload = json.loads(raw.decode("utf-8"))
                    self._apply_payload_to_ui(payload)
                    self._set_message("Keys loaded.")
                except Exception as e:
                    self._set_message(f"Error: {e}")

            def load_env_file(self):
                try:
                    from tkinter import Tk, filedialog
                    root = Tk()
                    root.withdraw()
                    env_path = filedialog.askopenfilename(
                        title="Select .env file",
                        filetypes=[("Env files", "*.env"), ("All files", "*.*")],
                    )
                    root.destroy()
                    if not env_path:
                        return
                    
                    payload = {}
                    for line in Path(env_path).read_text().splitlines():
                        line = line.strip()
                        if line and "=" in line and not line.startswith("#"):
                            k, v = line.split("=", 1)
                            payload[k.strip()] = v.strip().strip('"').strip("'")
                    
                    self._apply_payload_to_ui(payload)
                    self._set_message(f"Loaded from .env")
                except Exception as e:
                    self._set_message(f"Error: {e}")

            def bootstrap_keys(self):
                if self.config_path.exists():
                    try:
                        encrypted = self.config_path.read_bytes()
                        raw = _dpapi_decrypt(encrypted)
                        payload = json.loads(raw.decode("utf-8"))
                        self._apply_payload_to_ui(payload)
                    except Exception:
                        pass

        class KivyDesktopApp(App):
            title = "AI Trading System Desktop"

            def build(self):
                Builder.load_string(KV)
                return RootWidgetImpl()

            def on_stop(self):
                root = self.root
                if root:
                    root.stop_backend()

        _kivy_app_class = KivyDesktopApp
    
    return _kivy_app_class


class KivyDesktopApp:
    """Lazy-loading wrapper"""
    def run(self):
        AppClass = _get_kivy_app()
        AppClass().run()


if __name__ == "__main__":
    try:
        KivyDesktopApp().run()
    except Exception:
        error_text = traceback.format_exc()
        try:
            log_path = Path.home() / "AppData" / "Local" / "AITradingSystem" / "desktop_crash.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(error_text)
            ctypes.windll.user32.MessageBoxW(None, f"Crash: {log_path}", "Error", 0x10)
        except Exception:
            pass
        raise

