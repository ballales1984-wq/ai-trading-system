"""
AI Trading System - Desktop Application (Tkinter Version)
==========================================================
Applicazione desktop con Tkinter (incluso in Python).
"""

import os
import sys
import json
import webbrowser
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from pathlib import Path
from datetime import datetime

APP_NAME = "AI Trading System Pro"
APP_VERSION = "2.1.0"
CONFIG_FILE = Path.home() / ".ai_trading_config.json"


class ConfigManager:
    @staticmethod
    def load_config() -> dict:
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "binance_api_key": "",
            "binance_secret_key": "",
            "bybit_api_key": "",
            "bybit_secret_key": "",
            "testnet": True,
            "first_run": True
        }
    
    @staticmethod
    def save_config(config: dict) -> bool:
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception:
            return False
    
    @staticmethod
    def is_configured() -> bool:
        config = ConfigManager.load_config()
        return bool(config.get("binance_api_key")) and not config.get("first_run", True)


class SetupDialog:
    def __init__(self, parent):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"{APP_NAME} - Setup")
        self.dialog.geometry("500x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.build_ui()
    
    def build_ui(self):
        tk.Label(self.dialog, text=f"Benvenuto in {APP_NAME}", font=("Arial", 14, "bold")).pack(pady=10)
        tk.Label(self.dialog, text="Inserisci le tue API keys", font=("Arial", 10), fg="gray").pack(pady=5)
        
        form = tk.Frame(self.dialog)
        form.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(form, text="Binance API Key:").pack(anchor="w")
        self.binance_key = tk.Entry(form, show="*", width=40)
        self.binance_key.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(form, text="Binance Secret Key:").pack(anchor="w")
        self.binance_secret = tk.Entry(form, show="*", width=40)
        self.binance_secret.pack(fill=tk.X, pady=(0, 10))
        
        self.testnet_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self.dialog, text="Usa Testnet", variable=self.testnet_var).pack(pady=5)
        
        tk.Button(self.dialog, text="Salva", command=self.save, bg="#4CAF50", fg="white", width=15).pack(pady=10)
        
        tk.Label(self.dialog, text="Le chiavi sono salvate solo sul tuo PC", font=("Arial", 8), fg="gray").pack(pady=5)
    
    def save(self):
        config = {
            "binance_api_key": self.binance_key.get().strip(),
            "binance_secret_key": self.binance_secret.get().strip(),
            "testnet": self.testnet_var.get(),
            "first_run": False
        }
        
        if not config["binance_api_key"] or not config["binance_secret_key"]:
            messagebox.showerror("Errore", "Inserisci le API keys!")
            return
        
        if ConfigManager.save_config(config):
            messagebox.showinfo("OK", "Configurazione salvata!")
            self.dialog.destroy()
        else:
            messagebox.showerror("Errore", "Salvataggio fallito!")


class TradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.root.geometry("700x500")
        self.backend_process = None
        self.backend_running = False
        self.build_ui()
        
        if not ConfigManager.is_configured():
            self.root.after(100, self.show_setup)
    
    def build_ui(self):
        header = tk.Frame(self.root, bg="#2196F3", height=50)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text=f"{APP_NAME} - Control Panel", font=("Arial", 14, "bold"), bg="#2196F3", fg="white").pack(expand=True)
        
        main = tk.Frame(self.root, padx=20, pady=20)
        main.pack(fill=tk.BOTH, expand=True)
        
        self.status = tk.Label(main, text="Backend: Fermo", font=("Arial", 11), fg="red")
        self.status.pack(anchor="w", pady=(0, 10))
        
        btn_frame = tk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = tk.Button(btn_frame, text="Avvia Sistema", command=self.start_backend, bg="#4CAF50", fg="white", width=12)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_btn = tk.Button(btn_frame, text="Ferma", command=self.stop_backend, bg="#f44336", fg="white", width=12, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        tk.Button(btn_frame, text="Dashboard", command=self.open_dashboard, bg="#2196F3", fg="white", width=12).pack(side=tk.LEFT, padx=(0, 5))
        tk.Button(btn_frame, text="Config", command=self.show_setup, bg="#757575", fg="white", width=12).pack(side=tk.LEFT)
        
        tk.Label(main, text="Log:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 5))
        
        self.log_text = scrolledtext.ScrolledText(main, height=15, font=("Consolas", 9), bg="#1e1e1e", fg="#00ff00")
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(self.root, text=f"v{APP_VERSION}", font=("Arial", 8), fg="gray").pack(side=tk.BOTTOM, pady=5)
        
        self.log(f"{APP_NAME} v{APP_VERSION} started")
    
    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_text.see(tk.END)
    
    def show_setup(self):
        SetupDialog(self.root)
    
    def start_backend(self):
        if self.backend_running:
            return
        
        self.log("Starting backend...")
        config = ConfigManager.load_config()
        
        env = os.environ.copy()
        env["BINANCE_API_KEY"] = config.get("binance_api_key", "")
        env["BINANCE_SECRET_KEY"] = config.get("binance_secret_key", "")
        env["BINANCE_TESTNET"] = str(config.get("testnet", True)).lower()
        
        try:
            project_root = Path(__file__).parent.parent
            self.backend_process = subprocess.Popen(
                [sys.executable, "-m", "app.main"],
                cwd=str(project_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            
            threading.Thread(target=self.read_output, daemon=True).start()
            self.backend_running = True
            self.update_status()
            self.log("Backend started on http://localhost:8000")
        except Exception as e:
            self.log(f"Error: {e}")
    
    def read_output(self):
        if not self.backend_process:
            return
        for line in iter(self.backend_process.stdout.readline, ''):
            if line:
                self.root.after(0, lambda m=line.strip(): self.log(m))
    
    def stop_backend(self):
        if not self.backend_running or not self.backend_process:
            return
        
        self.log("Stopping backend...")
        try:
            self.backend_process.terminate()
            self.backend_process.wait(timeout=5)
        except:
            self.backend_process.kill()
        
        self.backend_running = False
        self.backend_process = None
        self.update_status()
        self.log("Backend stopped")
    
    def update_status(self):
        if self.backend_running:
            self.status.config(text="Backend: Attivo (http://localhost:8000)", fg="green")
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
        else:
            self.status.config(text="Backend: Fermo", fg="red")
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
    
    def open_dashboard(self):
        if self.backend_running:
            webbrowser.open("http://localhost:8000")
        else:
            webbrowser.open("https://ai-trading-system-kappa.vercel.app/dashboard")
    
    def on_closing(self):
        if self.backend_running and self.backend_process:
            self.stop_backend()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = TradingApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
