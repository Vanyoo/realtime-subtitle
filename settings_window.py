from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QLineEdit, 
                             QFormLayout, QPushButton, QComboBox, QDoubleSpinBox, QMessageBox)
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
        self.model_input = QComboBox()
        self.model_input.addItems(["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "claude-3-haiku-20240307", "groq-llama-3-70b"])
        self.model_input.setEditable(True) # Allow custom models
        self.model_input.setCurrentText(config.model)
        form_layout.addRow("Translation Model:", self.model_input)
        
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
        
        layout.addLayout(form_layout)
        
        # Save Button
        self.save_btn = QPushButton("Save & Restart")
        self.save_btn.clicked.connect(self.save_config)
        self.save_btn.setStyleSheet("background-color: #2ecc71; color: white; padding: 8px; border-radius: 4px;")
        layout.addWidget(self.save_btn)
        
        self.setLayout(layout)
        
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
        
        try:
            with open(config_path, 'w') as f:
                parser.write(f)
            QMessageBox.information(self, "Saved", "Configuration saved! The app should restart automatically.")
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save config: {e}")
