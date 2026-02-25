"""
AI Trading System - Desktop Application
======================================
Applicazione desktop completa per utenti paganti.
Include backend FastAPI + interfaccia Kivy.
"""

import os
import sys
import json
import webbrowser
import threading
import subprocess
from pathlib import Path

# Aggiungi il parent directory al path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from kivy.app import App
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.gridlayout import GridLayout
    from kivy.uix.label import Label
    from kivy.uix.textinput import TextInput
    from kivy.uix.button import Button
    from kivy.uix.popup import Popup
    from kivy.uix.progressbar import ProgressBar
    from kivy.uix.screenmanager import ScreenManager, Screen
    from kivy.core.window import Window
    from kivy.clock import Clock
    from kivy.properties import StringProperty, BooleanProperty
    from kivy.uix.widget import Widget
except ImportError:
    print("Kivy non installato. Installa con: pip install kivy")
    sys.exit(1)

# Configurazione
APP_NAME = "AI Trading System Pro"
APP_VERSION = "2.1.0"
CONFIG_FILE = Path.home() / ".ai_trading_config.json"


class ConfigManager:
    """Gestisce la configurazione utente (API keys, preferenze)."""
    
    @staticmethod
    def load_config() -> dict:
        """Carica configurazione da file."""
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
        """Salva configurazione su file."""
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            print(f"Errore salvataggio config: {e}")
            return False
    
    @staticmethod
    def is_configured() -> bool:
        """Verifica se l'app è già configurata."""
        config = ConfigManager.load_config()
        return bool(config.get("binance_api_key")) and not config.get("first_run", True)


class SetupScreen(Screen):
    """Schermata di setup iniziale per inserire API keys."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_ui()
    
    def build_ui(self):
        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
        
        # Header
        header = BoxLayout(size_hint_y=0.15)
        title = Label(
            text=f'[b]{APP_NAME}[/b]\nVersione {APP_VERSION}',
            markup=True,
            font_size='24sp',
            halign='center'
        )
        header.add_widget(title)
        layout.add_widget(header)
        
        # Descrizione
        desc = Label(
            text='Benvenuto! Inserisci le tue API keys per iniziare.\n'
                 'Le chiavi verranno salvate in modo sicuro sul tuo PC.',
            font_size='14sp',
            halign='center',
            size_hint_y=0.1
        )
        layout.add_widget(desc)
        
        # Form
        form_layout = GridLayout(cols=2, spacing=10, size_hint_y=0.5)
        
        # Binance API Key
        form_layout.add_widget(Label(text='Binance API Key:', size_hint_y=None, height=40))
        self.binance_key_input = TextInput(
            multiline=False,
            password=True,
            size_hint_y=None,
            height=40,
            hint_text='Inserisci la tua API key'
        )
        form_layout.add_widget(self.binance_key_input)
        
        # Binance Secret Key
        form_layout.add_widget(Label(text='Binance Secret Key:', size_hint_y=None, height=40))
        self.binance_secret_input = TextInput(
            multiline=False,
            password=True,
            size_hint_y=None,
            height=40,
            hint_text='Inserisci la tua Secret key'
        )
        form_layout.add_widget(self.binance_secret_input)
        
        # Bybit API Key (opzionale)
        form_layout.add_widget(Label(text='Bybit API Key (opzionale):', size_hint_y=None, height=40))
        self.bybit_key_input = TextInput(
            multiline=False,
            password=True,
            size_hint_y=None,
            height=40,
            hint_text='Opzionale'
        )
        form_layout.add_widget(self.bybit_key_input)
        
        # Bybit Secret Key (opzionale)
        form_layout.add_widget(Label(text='Bybit Secret Key (opzionale):', size_hint_y=None, height=40))
        self.bybit_secret_input = TextInput(
            multiline=False,
            password=True,
            size_hint_y=None,
            height=40,
            hint_text='Opzionale'
        )
        form_layout.add_widget(self.bybit_secret_input)
        
        layout.add_widget(form_layout)
        
        # Testnet toggle
        self.testnet_btn = Button(
            text='Usa Testnet (consigliato per iniziare)',
            size_hint_y=None,
            height=50,
            background_color=(0.2, 0.8, 0.2, 1)
        )
        self.testnet_btn.bind(on_press=self.toggle_testnet)
        self.testnet = True
        layout.add_widget(self.testnet_btn)
        
        # Spacer
        layout.add_widget(Widget(size_hint_y=0.1))
        
        # Bottone salva
        save_btn = Button(
            text='Salva e Continua',
            size_hint_y=None,
            height=60,
            font_size='18sp',
            background_color=(0.2, 0.6, 1, 1)
        )
        save_btn.bind(on_press=self.save_config)
        layout.add_widget(save_btn)
        
        # Note sicurezza
        security_note = Label(
            text='Le tue API keys sono salvate solo sul tuo PC\n'
                 'e mai trasmesse a server esterni.',
            font_size='12sp',
            halign='center',
            size_hint_y=0.1,
            color=(0.5, 0.5, 0.5, 1)
        )
        layout.add_widget(security_note)
        
        self.add_widget(layout)
    
    def toggle_testnet(self, instance):
        self.testnet = not self.testnet
        if self.testnet:
            instance.text = 'Usa Testnet (consigliato per iniziare)'
            instance.background_color = (0.2, 0.8, 0.2, 1)
        else:
            instance.text = 'Usa Mainnet (soldi reali)'
            instance.background_color = (0.8, 0.2, 0.2, 1)
    
    def save_config(self, instance):
        config = {
            "binance_api_key": self.binance_key_input.text.strip(),
            "binance_secret_key": self.binance_secret_input.text.strip(),
            "bybit_api_key": self.bybit_key_input.text.strip(),
            "bybit_secret_key": self.bybit_secret_input.text.strip(),
            "testnet": self.testnet,
            "first_run": False
        }
        
        # Validazione
        if not config["binance_api_key"] or not config["binance_secret_key"]:
            self.show_error("Inserisci almeno le API keys di Binance!")
            return
        
        # Salva
        if ConfigManager.save_config(config):
            self.show_success("Configurazione salvata!")
            Clock.schedule_once(lambda dt: self.switch_to_main(), 1.5)
        else:
            self.show_error("Errore nel salvataggio!")
    
    def show_error(self, message):
        popup = Popup(
            title='Errore',
            content=Label(text=message),
            size_hint=(None, None),
            size=(400, 200)
        )
        popup.open()
    
    def show_success(self, message):
        popup = Popup(
            title='Successo',
            content=Label(text=message, color=(0, 1, 0, 1)),
            size_hint=(None, None),
            size=(400, 200)
        )
        popup.open()
    
    def switch_to_main(self):
        self.manager.current = 'main'


class MainScreen(Screen):
    """Schermata principale con controllo del sistema."""
    
    backend_running = BooleanProperty(False)
    backend_url = StringProperty("")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backend_process = None
        self.build_ui()
    
    def build_ui(self):
        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
        
        # Header
        header = BoxLayout(size_hint_y=0.1)
        title = Label(
            text=f'[b]{APP_NAME}[/b] - Pannello di Controllo',
            markup=True,
            font_size='20sp',
            halign='center'
        )
        header.add_widget(title)
        layout.add_widget(header)
        
        # Status
        self.status_label = Label(
            text='Backend: Fermo',
            font_size='16sp',
            size_hint_y=0.1,
            color=(1, 0, 0, 1)
        )
        layout.add_widget(self.status_label)
        
        # Progress bar (nascosta di default)
        self.progress = ProgressBar(max=100, value=0, size_hint_y=None, height=20)
        self.progress.opacity = 0
        layout.add_widget(self.progress)
        
        # Bottoni controllo
        buttons_layout = BoxLayout(orientation='horizontal', spacing=10, size_hint_y=0.15)
        
        self.start_btn = Button(
            text='Avvia Sistema',
            font_size='16sp',
            background_color=(0.2, 0.8, 0.2, 1)
        )
        self.start_btn.bind(on_press=self.start_backend)
        buttons_layout.add_widget(self.start_btn)
        
        self.stop_btn = Button(
            text='Ferma Sistema',
            font_size='16sp',
            background_color=(0.8, 0.2, 0.2, 1),
            disabled=True
        )
        self.stop_btn.bind(on_press=self.stop_backend)
        buttons_layout.add_widget(self.stop_btn)
        
        layout.add_widget(buttons_layout)
        
        # Bottoni azione
        actions_layout = BoxLayout(orientation='horizontal', spacing=10, size_hint_y=0.15)
        
        dashboard_btn = Button(
            text='Apri Dashboard',
            font_size='14sp',
            background_color=(0.2, 0.6, 1, 1)
        )
        dashboard_btn.bind(on_press=self.open_dashboard)
        actions_layout.add_widget(dashboard_btn)
        
        config_btn = Button(
            text='Modifica Config',
            font_size='14sp',
            background_color=(0.6, 0.6, 0.6, 1)
        )
        config_btn.bind(on_press=self.edit_config)
        actions_layout.add_widget(config_btn)
        
        layout.add_widget(actions_layout)
        
        # Log area
        log_label = Label(
            text='Log Sistema:',
            font_size='14sp',
            size_hint_y=None,
            height=30,
            halign='left'
        )
        layout.add_widget(log_label)
        
        self.log_text = TextInput(
            multiline=True,
            readonly=True,
            font_size='12sp',
            background_color=(0.1, 0.1, 0.1, 1),
            foreground_color=(0, 1, 0, 1),
            size_hint_y=0.4
        )
        layout.add_widget(self.log_text)
        
        # Info
        info = Label(
            text=f'v{APP_VERSION} | Config: {CONFIG_FILE}',
            font_size='10sp',
            size_hint_y=None,
            height=20,
            color=(0.5, 0.5, 0.5, 1)
        )
        layout.add_widget(info)
        
        self.add_widget(layout)
    
    def log(self, message):
        """Aggiunge messaggio al log."""
        self.log_text.text += f"{message}\n"
        self.log_text.cursor = (0, len(self.log_text.text))
    
    def start_backend(self, instance):
        """Avvia il backend FastAPI."""
        if self.backend_running:
            return
        
        self.log("Avvio backend...")
        self.progress.opacity = 1
        self.progress.value = 0
        
        # Carica config
        config = ConfigManager.load_config()
        
        # Imposta variabili d'ambiente
        env = os.environ.copy()
        env["BINANCE_API_KEY"] = config.get("binance_api_key", "")
        env["BINANCE_SECRET_KEY"] = config.get("binance_secret_key", "")
        env["BYBIT_API_KEY"] = config.get("bybit_api_key", "")
        env["BYBIT_SECRET_KEY"] = config.get("bybit_secret_key", "")
        env["BINANCE_TESTNET"] = str(config.get("testnet", True)).lower()
        
        try:
            # Avvia backend
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
            
            # Thread per leggere output
            threading.Thread(target=self.read_backend_output, daemon=True).start()
            
            self.backend_running = True
            self.backend_url = "http://localhost:8000"
            self.update_status()
            
            # Simula progresso
            Clock.schedule_interval(self.update_progress, 0.1)
            
            self.log("Backend avviato su http://localhost:8000")
            
        except Exception as e:
            self.log(f"Errore avvio backend: {e}")
            self.progress.opacity = 0
    
    def read_backend_output(self):
        """Legge l'output del backend."""
        if not self.backend_process:
            return
        
        for line in iter(self.backend_process.stdout.readline, ''):
            if line:
                Clock.schedule_once(lambda dt, msg=line.strip(): self.log(msg), 0)
    
    def update_progress(self, dt):
        """Aggiorna la progress bar."""
        if self.progress.value < 90:
            self.progress.value += 2
        return self.backend_running
    
    def stop_backend(self, instance):
        """Ferma il backend."""
        if not self.backend_running or not self.backend_process:
            return
        
        self.log("Arresto backend...")
        
        try:
            self.backend_process.terminate()
            self.backend_process.wait(timeout=5)
        except:
            self.backend_process.kill()
        
        self.backend_running = False
        self.backend_process = None
        self.progress.value = 0
        self.progress.opacity = 0
        
        self.update_status()
        self.log("Backend fermato")
    
    def update_status(self):
        """Aggiorna lo stato UI."""
        if self.backend_running:
            self.status_label.text = f'Backend: Attivo ({self.backend_url})'
            self.status_label.color = (0, 1, 0, 1)
            self.start_btn.disabled = True
            self.stop_btn.disabled = False
        else:
            self.status_label.text = 'Backend: Fermo'
            self.status_label.color = (1, 0, 0, 1)
            self.start_btn.disabled = False
            self.stop_btn.disabled = True
    
    def open_dashboard(self, instance):
        """Apre la dashboard nel browser."""
        if self.backend_running:
            webbrowser.open("http://localhost:8000")
        else:
            # Apre la versione online
            webbrowser.open("https://ai-trading-system-kappa.vercel.app/dashboard")
    
    def edit_config(self, instance):
        """Torna alla schermata di setup."""
        self.manager.current = 'setup'


class TradingApp(App):
    """Applicazione principale."""
    
    def build(self):
        Window.title = f"{APP_NAME} v{APP_VERSION}"
        Window.size = (800, 600)
        
        # Screen manager
        sm = ScreenManager()
        
        # Controlla se è il primo avvio
        if ConfigManager.is_configured():
            sm.add_widget(MainScreen(name='main'))
            sm.add_widget(SetupScreen(name='setup'))
            sm.current = 'main'
        else:
            sm.add_widget(SetupScreen(name='setup'))
            sm.add_widget(MainScreen(name='main'))
            sm.current = 'setup'
        
        return sm
    
    def on_stop(self):
        """Pulizia alla chiusura."""
        if hasattr(self.root.get_screen('main'), 'backend_process'):
            main_screen = self.root.get_screen('main')
            if main_screen.backend_process:
                main_screen.backend_process.terminate()


if __name__ == '__main__':
    TradingApp().run()
