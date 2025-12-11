from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QLineEdit, QHBoxLayout,
                             QFormLayout, QPushButton, QComboBox, QDoubleSpinBox, QMessageBox, QSpinBox)
from PyQt6.QtCore import Qt
import configparser
import os
from config import config

class SettingsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Settings")
        self.setFixedSize(400, 300)
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        form_layout = QFormLayout()
        
        # API Key
        self.api_key_input = QLineEdit()
        self.api_key_input.setText(config.api_key if config.api_key != "dummy-key-for-local" else "")
        self.api_key_input.setPlaceholderText("sk-...")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        form_layout.addRow("OpenAI API Key:", self.api_key_input)
        
        # Base URL
        self.base_url_input = QLineEdit()
        self.base_url_input.setText(config.api_base_url if config.api_base_url else "")
        self.base_url_input.setPlaceholderText("https://api.openai.com/v1")
        form_layout.addRow("API Base URL:", self.base_url_input)
        
        # Translation Model
        # Translation Model
        self.model_input = QComboBox()
        self.model_input.setEditable(True) # Allow custom models
        # Initial list just has current model
        initial_models = [config.model] if config.model else ["gpt-3.5-turbo"]
        self.model_input.addItems(initial_models)
        self.model_input.setCurrentText(config.model)
        
        # Container for Model + Refresh
        model_layout = QVBoxLayout()
        model_row = extract_row = QWidget()
        model_row_layout = QHBoxLayout(model_row)
        model_row_layout.setContentsMargins(0,0,0,0)
        model_row_layout.addWidget(self.model_input)
        
        self.refresh_btn = QPushButton("🔄")
        self.refresh_btn.setToolTip("Fetch models from server")
        self.refresh_btn.setFixedWidth(30)
        self.refresh_btn.clicked.connect(self.fetch_models)
        model_row_layout.addWidget(self.refresh_btn)
        
        form_layout.addRow("Translation Model:", model_row)
        
        # Whisper Model
        self.whisper_input = QComboBox()
        self.whisper_input.addItems(["tiny", "base", "small", "medium", "large", "turbo"])
        self.whisper_input.setCurrentText(config.whisper_model)
        form_layout.addRow("Whisper Model:", self.whisper_input)
        
        # Streaming Step Size (Latency vs CPU)
        self.step_size_input = QDoubleSpinBox()
        self.step_size_input.setRange(0.1, 2.0)
        self.step_size_input.setSingleStep(0.1)
        self.step_size_input.setValue(config.streaming_step_size)
        form_layout.addRow("Stream Step (s):", self.step_size_input)
        
        # Silence Duration (VAD)
        self.silence_dur_input = QDoubleSpinBox()
        self.silence_dur_input.setRange(0.2, 5.0)
        self.silence_dur_input.setSingleStep(0.1)
        self.silence_dur_input.setValue(config.silence_duration)
        self.silence_dur_input.setSuffix(" s")
        form_layout.addRow("Silence VAD (s):", self.silence_dur_input)

        # Max Phrase Duration
        self.max_phrase_input = QSpinBox()
        self.max_phrase_input.setRange(5, 600) # 5s to 10m
        self.max_phrase_input.setSingleStep(5)
        self.max_phrase_input.setValue(int(config.max_phrase_duration))
        self.max_phrase_input.setSuffix(" s")
        form_layout.addRow("Max Phrase (s):", self.max_phrase_input)
        
        layout.addLayout(form_layout)
        
        # Save Button
        self.save_btn = QPushButton("Save & Restart")
        self.save_btn.clicked.connect(self.save_config)
        self.save_btn.setStyleSheet("background-color: #2ecc71; color: white; padding: 8px; border-radius: 4px;")
        layout.addWidget(self.save_btn)
        
        self.setLayout(layout)
        
    def fetch_models(self):
        """Fetch models from the provided Base URL using openai client"""
        from openai import OpenAI
        
        api_key = self.api_key_input.text() or "dummy"
        base_url = self.base_url_input.text()
        
        if not base_url:
            QMessageBox.warning(self, "Missing URL", "Please enter an API Base URL first.")
            return

        self.refresh_btn.setEnabled(False)
        self.model_input.clear()
        self.model_input.addItem("Fetching...")
        
        # Use a thread to avoid freezing UI
        import threading
        def _fetch():
            try:
                client = OpenAI(api_key=api_key, base_url=base_url)
                models_response = client.models.list()
                # Sort models by id
                model_ids = sorted([m.id for m in models_response.data])
                
                # Update UI in main thread safety check? 
                # PyQt technically requires signal for thread safety, but for simple list update...
                # Let's verify if we need signals. Yes strictly speaking.
                # Use QTimer to update on main thread or simple hack?
                # For safety, I will just call a reactor method or use strict signals. 
                # Actually, to save complexity in this small edit, I'll run sync with a short timeout?
                # No, freezing is bad. I'll use a simple QTimer check or try/except.
                # Wait, I can't easily add signals now without major refactor.
                # I will run it synchronously with a strict timeout of 3s, assuming user server is local or fast.
                pass 
            except Exception as e:
                pass

        # REVISING: Threading without signals in PyQt will crash if we touch UI.
        # Running sync is safer for now, user expects a pause with "Fetch".
        
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            # 3 second timeout to prevent long hang
            models_response = client.models.list(timeout=5.0) 
            model_ids = sorted([m.id for m in models_response.data])
            
            self.model_input.clear()
            self.model_input.addItems(model_ids)
            
            # Restore current if in list
            current = config.model
            if current in model_ids:
                self.model_input.setCurrentText(current)
            elif model_ids:
                self.model_input.setCurrentIndex(0)
                
            QMessageBox.information(self, "Success", f"Found {len(model_ids)} models.")
            
        except Exception as e:
            self.model_input.clear()
            self.model_input.addItem(config.model) # Restore
            QMessageBox.critical(self, "Error", f"Failed to fetch models: {str(e)}")
            
        self.refresh_btn.setEnabled(True)
        
    def save_config(self):
        """Write to config.ini"""
        config_path = os.path.join(os.path.dirname(__file__), "config.ini")
        parser = configparser.ConfigParser()
        parser.read(config_path)
        
        # Update values
        if not parser.has_section("api"): parser.add_section("api")
        if not parser.has_section("translation"): parser.add_section("translation")
        if not parser.has_section("transcription"): parser.add_section("transcription")
        if not parser.has_section("audio"): parser.add_section("audio")
        
        parser.set("api", "api_key", self.api_key_input.text() or "")
        parser.set("api", "base_url", self.base_url_input.text() or "")
        parser.set("translation", "model", self.model_input.currentText())
        parser.set("transcription", "whisper_model", self.whisper_input.currentText())
        parser.set("audio", "streaming_step_size", str(self.step_size_input.value()))
        parser.set("audio", "max_phrase_duration", str(self.max_phrase_input.value()))
        parser.set("audio", "silence_duration", str(self.silence_dur_input.value()))
        
        try:
            with open(config_path, 'w') as f:
                parser.write(f)
            QMessageBox.information(self, "Saved", "Configuration saved! The app should restart automatically.")
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save config: {e}")
