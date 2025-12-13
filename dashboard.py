from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QComboBox, QLineEdit, 
                             QTabWidget, QSpinBox, QDoubleSpinBox, QGridLayout,
                             QScrollArea, QSizePolicy, QSpacerItem, QFormLayout, QApplication,
                             QMessageBox, QTextEdit, QDialog)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QIcon, QColor
import sys
import sounddevice as sd
from config import config

# Modern QSS Styles
STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Helvetica Neue', 'Segoe UI', Arial, sans-serif;
}
QTabWidget::pane {
    border: 1px solid #313244;
    background: #1e1e2e;
    border-radius: 8px;
}
QTabBar::tab {
    background: #313244;
    color: #a6adc8;
    padding: 10px 20px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #89b4fa;
    color: #1e1e2e;
    font-weight: bold;
}
QLabel {
    font-size: 14px;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 5px;
    color: #cdd6f4;
    selection-background-color: #585b70;
}
QPushButton {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #b4befe;
}
QPushButton#StopButton {
    background-color: #f38ba8;
}
QPushButton#StopButton:hover {
    background-color: #eba0ac;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    color: #fab387;
}
"""

class Dashboard(QWidget):
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def closeEvent(self, event):
        """Ensure total program quit when dashboard is closed"""
        self.status_label.setText("Stopping...")
        self.on_stop()
        # Force application exit
        QApplication.quit()
        event.accept()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Translator - Control Center")
        self.setMinimumSize(600, 500)
        self.setStyleSheet(STYLESHEET)
        
        # Main Layout
        self.layout = QVBoxLayout()
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(self.layout)
        
        # Header
        header = QLabel("🎙️ Real-Time Translator")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #89b4fa;")
        self.layout.addWidget(header)
        
        # Tabs
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        
        self.init_home_tab()
        self.init_audio_tab()
        self.init_device_manager_tab()
        self.init_transcription_tab()
        self.init_translation_tab()
        
        # Footer Actions
        footer = QHBoxLayout()
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.clicked.connect(self.save_config)
        self.save_btn.setStyleSheet("""
            background-color: #a6e3a1; color: #1e1e2e;
        """)
        footer.addStretch()
        footer.addWidget(self.save_btn)
        self.layout.addLayout(footer)

    def init_home_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-size: 18px; color: #a6e3a1;")
        layout.addWidget(self.status_label)
        
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ Launch Translator")
        self.start_btn.setFixedSize(200, 60)
        self.start_btn.setStyleSheet("font-size: 16px; background-color: #89b4fa; border-radius: 10px;")
        self.start_btn.clicked.connect(self.on_start)
        
        self.stop_btn = QPushButton("⏹ Stop Translator")
        self.stop_btn.setFixedSize(200, 60)
        self.stop_btn.setStyleSheet("font-size: 16px; background-color: #f38ba8; border-radius: 10px;")
        self.stop_btn.clicked.connect(self.on_stop)
        self.stop_btn.hide()
        
        layout.addLayout(btn_layout)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        
        info = QLabel("The translator will open as an overlay window.\nYou can minimize this dashboard.")
        info.setStyleSheet("color: #6c7086; font-style: italic;")
        layout.addWidget(info)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "🏠 Home")

    def init_audio_tab(self):
        tab = QWidget()
        layout = QGridLayout() # Use Grid for organized form
        layout.setSpacing(15)
        
        # Device Selection
        layout.addWidget(QLabel("Input Device:"), 0, 0)
        self.device_combo = QComboBox()
        self.populate_devices()
        layout.addWidget(self.device_combo, 0, 1)
        
        # Refresh Button
        refresh_btn = QPushButton("🔄")
        refresh_btn.setFixedWidth(40)
        refresh_btn.clicked.connect(self.populate_devices)
        layout.addWidget(refresh_btn, 0, 2)
        
        # Sample Rate
        layout.addWidget(QLabel("Sample Rate:"), 1, 0)
        self.sample_rate = QSpinBox()
        self.sample_rate.setRange(8000, 48000)
        self.sample_rate.setValue(config.sample_rate)
        layout.addWidget(self.sample_rate, 1, 1)

        # Silence Threshold
        layout.addWidget(QLabel("Silence Threshold:"), 2, 0)
        self.silence_thresh = QDoubleSpinBox()
        self.silence_thresh.setRange(0.001, 1.0)
        self.silence_thresh.setSingleStep(0.001)
        self.silence_thresh.setDecimals(3)
        self.silence_thresh.setValue(config.silence_threshold)
        layout.addWidget(self.silence_thresh, 2, 1)
        
        layout.addWidget(QLabel("Silence Duration (s):"), 3, 0)
        self.silence_dur = QDoubleSpinBox()
        self.silence_dur.setValue(config.silence_duration)
        layout.addWidget(self.silence_dur, 3, 1)
        
        layout.setRowStretch(4, 1) # Push to top
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "🎤 Audio")

    def init_device_manager_tab(self):
        """Audio Device Manager - Create/Manage Multi-Output Devices"""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # Header
        header = QLabel("Audio Device Manager")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #fab387;")
        layout.addWidget(header)
        
        info = QLabel("Create multi-output devices to capture system audio + hear it through speakers")
        info.setStyleSheet("color: #6c7086; font-size: 12px; font-style: italic;")
        layout.addWidget(info)
        
        # Available Devices List
        devices_label = QLabel("Available Output Devices:")
        layout.addWidget(devices_label)
        
        self.output_devices_list = QComboBox()
        self.output_devices_list.setMinimumHeight(30)
        layout.addWidget(self.output_devices_list)
        
        # Virtual Device List
        virtual_label = QLabel("Virtual/BlackHole Devices:")
        layout.addWidget(virtual_label)
        
        self.virtual_devices_list = QComboBox()
        self.virtual_devices_list.setMinimumHeight(30)
        layout.addWidget(self.virtual_devices_list)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.refresh_devices_btn = QPushButton("🔄 Refresh Devices")
        self.refresh_devices_btn.clicked.connect(self.refresh_audio_devices)
        btn_layout.addWidget(self.refresh_devices_btn)
        
        self.create_multi_output_btn = QPushButton("➕ Create Multi-Output Device")
        self.create_multi_output_btn.setStyleSheet("""
            background-color: #a6e3a1; color: #1e1e2e; font-weight: bold;
        """)
        self.create_multi_output_btn.clicked.connect(self.create_multi_output_device)
        btn_layout.addWidget(self.create_multi_output_btn)
        
        layout.addLayout(btn_layout)
        
        # Set as Default Button
        self.set_default_btn = QPushButton("🔊 Set Selected as Default Output")
        self.set_default_btn.clicked.connect(self.set_default_output_device)
        layout.addWidget(self.set_default_btn)
        
        # Status
        self.device_status = QLabel("Ready")
        self.device_status.setStyleSheet("color: #a6e3a1; font-style: italic; padding: 10px;")
        layout.addWidget(self.device_status)
        
        # Help text
        help_text = QLabel(
            "<b>How to use:</b><br>"
            "1. Select your speakers from 'Available Output Devices'<br>"
            "2. Select BlackHole from 'Virtual Devices'<br>"
            "3. Click 'Create Multi-Output Device'<br>"
            "   • Audio MIDI Setup will open with instructions<br>"
            "   • Follow the step-by-step guide in the terminal/console<br>"
            "4. The new device lets you hear audio AND capture it!<br>"
            "<br><i>Note: Accessibility permissions may be required for automation.<br>"
            "Without permissions, you'll see manual instructions (very easy!).</i>"
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("background-color: #313244; padding: 10px; border-radius: 5px; font-size: 12px;")
        layout.addWidget(help_text)
        
        layout.addStretch()
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "🔧 Device Manager")
        
        # Initial population
        self.refresh_audio_devices()

    def refresh_audio_devices(self):
        """Refresh the list of audio devices"""
        try:
            import platform
            if platform.system() != "Darwin":
                self.device_status.setText("⚠️ Device Manager only available on macOS")
                self.device_status.setStyleSheet("color: #fab387;")
                return
            
            from audio_device_manager import AudioDeviceManager
            manager = AudioDeviceManager()
            
            # Get output devices
            output_devices = manager.get_output_devices()
            self.output_devices_list.clear()
            for device in output_devices:
                self.output_devices_list.addItem(f"{device['name']}", device['id'])
            
            # Get virtual/BlackHole devices
            virtual_devices = manager.get_virtual_devices()
            self.virtual_devices_list.clear()
            if not virtual_devices:
                self.virtual_devices_list.addItem("No BlackHole device found - Please install it")
                self.device_status.setText("⚠️ BlackHole not found. Install: brew install blackhole-2ch")
                self.device_status.setStyleSheet("color: #fab387;")
            else:
                for device in virtual_devices:
                    self.virtual_devices_list.addItem(f"{device['name']}", device['id'])
                self.device_status.setText("✅ Devices loaded successfully")
                self.device_status.setStyleSheet("color: #a6e3a1;")
                
        except ImportError:
            self.device_status.setText("⚠️ Audio device management requires PyObjC (pip install pyobjc-framework-CoreAudio)")
            self.device_status.setStyleSheet("color: #f38ba8;")
        except Exception as e:
            self.device_status.setText(f"❌ Error: {str(e)}")
            self.device_status.setStyleSheet("color: #f38ba8;")
    
    def create_multi_output_device(self):
        """Create a multi-output device combining speakers + BlackHole"""
        try:
            from audio_device_manager import AudioDeviceManager
            manager = AudioDeviceManager()
            
            output_device_id = self.output_devices_list.currentData()
            virtual_device_id = self.virtual_devices_list.currentData()
            
            if not output_device_id or not virtual_device_id:
                self.device_status.setText("⚠️ Please select both devices")
                self.device_status.setStyleSheet("color: #fab387;")
                return
            
            # Show instruction dialog
            self._show_multi_output_instructions()
            
            # Call the audio device manager to open Audio MIDI Setup
            device_name = f"Translator Multi-Output"
            success = manager.create_multi_output_device(
                device_name,
                [output_device_id, virtual_device_id],
                silent=True  # Suppress console output, show GUI dialog instead
            )
            
            if success:
                self.device_status.setText(f"✅ Audio MIDI Setup opened - Follow the instructions!")
                self.device_status.setStyleSheet("color: #a6e3a1;")
                # Refresh after user has time to create the device
                QTimer = __import__('PyQt6.QtCore', fromlist=['QTimer']).QTimer
                QTimer.singleShot(3000, self.refresh_audio_devices)
            else:
                self.device_status.setText("❌ Failed to open Audio MIDI Setup")
                self.device_status.setStyleSheet("color: #f38ba8;")
                
        except Exception as e:
            self.device_status.setText(f"❌ Error: {str(e)}")
            self.device_status.setStyleSheet("color: #f38ba8;")
    
    def _show_multi_output_instructions(self):
        """Show a dialog with step-by-step instructions"""
        dialog = QDialog(self)
        dialog.setWindowTitle("🎵 Create Multi-Output Device - Instructions")
        dialog.setMinimumSize(600, 500)
        dialog.setStyleSheet(STYLESHEET)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("📋 Step-by-Step Guide")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #89b4fa; padding: 10px;")
        layout.addWidget(title)
        
        # Instructions text
        instructions = QTextEdit()
        instructions.setReadOnly(True)
        instructions.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 8px;
                padding: 15px;
                font-size: 13px;
                line-height: 1.6;
            }
        """)
        
        output_device = self.output_devices_list.currentText()
        virtual_device = self.virtual_devices_list.currentText()
        
        instructions_html = f"""
        <div style='font-family: Arial, sans-serif;'>
        <h3 style='color: #fab387;'>✨ Audio MIDI Setup is opening...</h3>
        
        <p style='color: #a6adc8;'><b>Follow these simple steps:</b></p>
        
        <div style='background: #313244; padding: 12px; border-radius: 6px; margin: 10px 0;'>
        <p style='color: #89b4fa; font-weight: bold;'>👉 Step 1: Find the Plus Button</p>
        <p>In the Audio MIDI Setup window, look at the <b>bottom-left corner</b>.<br>
        Click the <span style='background: #45475a; padding: 2px 8px; border-radius: 3px;'>[+]</span> button.</p>
        </div>
        
        <div style='background: #313244; padding: 12px; border-radius: 6px; margin: 10px 0;'>
        <p style='color: #89b4fa; font-weight: bold;'>👉 Step 2: Create Multi-Output</p>
        <p>From the menu that appears, select:<br>
        <span style='color: #a6e3a1; font-weight: bold;'>“Create Multi-Output Device”</span></p>
        </div>
        
        <div style='background: #313244; padding: 12px; border-radius: 6px; margin: 10px 0;'>
        <p style='color: #89b4fa; font-weight: bold;'>👉 Step 3: Select Devices</p>
        <p>Check the boxes for these devices:<br>
        ✅ <span style='color: #f9e2af;'>{output_device}</span> (your speakers)<br>
        ✅ <span style='color: #f9e2af;'>{virtual_device}</span> (for capturing)</p>
        </div>
        
        <div style='background: #313244; padding: 12px; border-radius: 6px; margin: 10px 0;'>
        <p style='color: #89b4fa; font-weight: bold;'>👉 Step 4: Configure Drift Correction</p>
        <p><b style='color: #f38ba8;'>IMPORTANT:</b> Uncheck <b>“Drift Correction”</b> for <span style='color: #f9e2af;'>{output_device}</span><br>
        (This allows you to hear the audio through your speakers)</p>
        </div>
        
        <div style='background: #313244; padding: 12px; border-radius: 6px; margin: 10px 0;'>
        <p style='color: #89b4fa; font-weight: bold;'>👉 Step 5: Set as Default Output</p>
        <p>Go to <b>System Settings → Sound</b><br>
        Set the new <span style='color: #a6e3a1;'>Multi-Output Device</span> as your output device.</p>
        </div>
        
        <hr style='border: 1px solid #45475a; margin: 15px 0;'>
        
        <p style='color: #6c7086; font-style: italic;'>
        💡 <b>Tip:</b> You only need to do this once! The device will persist across reboots.<br>
        After setup, you'll hear audio normally while the translator captures it in real-time.
        </p>
        </div>
        """
        
        instructions.setHtml(instructions_html)
        layout.addWidget(instructions)
        
        # Close button
        close_btn = QPushButton("✅ Got it!")
        close_btn.setFixedHeight(40)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1;
                color: #1e1e2e;
                font-weight: bold;
                font-size: 14px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #b4e4b4;
            }
        """)
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec()
    
    def set_default_output_device(self):
        """Set the selected device as system default output"""
        try:
            from audio_device_manager import AudioDeviceManager
            manager = AudioDeviceManager()
            
            device_id = self.output_devices_list.currentData()
            if not device_id:
                self.device_status.setText("⚠️ Please select a device")
                self.device_status.setStyleSheet("color: #fab387;")
                return
            
            device_name = self.output_devices_list.currentText()
            success = manager.set_default_output_device(device_id)
            
            if success:
                self.device_status.setText(f"✅ Set '{device_name}' as default output")
                self.device_status.setStyleSheet("color: #a6e3a1;")
            else:
                self.device_status.setText("❌ Failed to set default device")
                self.device_status.setStyleSheet("color: #f38ba8;")
                
        except Exception as e:
            self.device_status.setText(f"❌ Error: {str(e)}")
            self.device_status.setStyleSheet("color: #f38ba8;")

    def refresh_model_list(self):
        """Fetch available models from the API and populate the model dropdown"""
        try:
            from openai import OpenAI
            import httpx
            
            api_key = self.api_key.text() or "dummy-key-for-local"
            base_url = self.base_url.text() or None
            
            # Update button state
            self.refresh_models_btn.setEnabled(False)
            self.refresh_models_btn.setText("...")
            
            # Create client with SSL verification disabled
            http_client = httpx.Client(verify=False)
            client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
            
            # Fetch models
            models_response = client.models.list()
            model_ids = [model.id for model in models_response.data]
            
            # Update combo box
            current_model = self.model.currentText()
            self.model.clear()
            
            if model_ids:
                self.model.addItems(sorted(model_ids))
                # Try to restore previous selection
                index = self.model.findText(current_model)
                if index >= 0:
                    self.model.setCurrentIndex(index)
                    
                # Show success in status label if we're on the home tab
                if hasattr(self, 'status_label'):
                    self.status_label.setText(f"✅ Loaded {len(model_ids)} models")
                    self.status_label.setStyleSheet("font-size: 18px; color: #a6e3a1;")
            else:
                self.model.addItem(current_model)
                if hasattr(self, 'status_label'):
                    self.status_label.setText("⚠️ No models found")
                    self.status_label.setStyleSheet("font-size: 18px; color: #fab387;")
            
        except Exception as e:
            # Restore original model on error
            if not self.model.currentText():
                self.model.addItem(config.model)
            
            error_msg = str(e)
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"❌ Failed to fetch models: {error_msg[:50]}")
                self.status_label.setStyleSheet("font-size: 18px; color: #f38ba8;")
            print(f"[Dashboard] Model refresh error: {error_msg}")
        
        finally:
            # Restore button state
            self.refresh_models_btn.setEnabled(True)
            self.refresh_models_btn.setText("🔄")

    def init_transcription_tab(self):
        tab = QWidget()
        layout = QFormLayout()
        
        self.whisper_model = QComboBox()
        self.whisper_model.addItems(["tiny", "base", "small", "medium", "large-v3", "turbo"])
        self.whisper_model.setCurrentText(config.whisper_model)
        layout.addRow("Whisper Model:", self.whisper_model)
        
        self.device_type = QComboBox()
        self.device_type.addItems(["cpu", "cuda", "mps", "auto"])
        self.device_type.setCurrentText(config.whisper_device)
        layout.addRow("Compute Device:", self.device_type)
        
        self.compute_type = QComboBox()
        self.compute_type.addItems(["int8", "float16", "float32"])
        self.compute_type.setCurrentText(config.whisper_compute_type)
        layout.addRow("Quantization:", self.compute_type)
        
        # Source Language Configuration
        self.source_language = QComboBox()
        self.source_language.setEditable(True)
        self.source_language.addItems(["auto", "en", "zh", "vi", "ja", "ko", "es", "fr", "de", "ru", "ar", "pt", "it"])
        source_lang = config.source_language if config.source_language else "auto"
        self.source_language.setCurrentText(source_lang)
        layout.addRow("Source Language:", self.source_language)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "📝 Transcription")

    def init_translation_tab(self):
        tab = QWidget()
        layout = QFormLayout()
        
        self.api_key = QLineEdit(config.api_key)
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("sk-...")
        layout.addRow("API Key:", self.api_key)
        
        self.base_url = QLineEdit(config.api_base_url or "")
        self.base_url.setPlaceholderText("https://api.openai.com/v1")
        layout.addRow("Base URL:", self.base_url)
        
        # Model selection with refresh button
        model_layout = QHBoxLayout()
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.addItem(config.model)
        model_layout.addWidget(self.model)
        
        self.refresh_models_btn = QPushButton("🔄")
        self.refresh_models_btn.setFixedWidth(40)
        self.refresh_models_btn.setToolTip("Refresh model list from API")
        self.refresh_models_btn.clicked.connect(self.refresh_model_list)
        model_layout.addWidget(self.refresh_models_btn)
        
        layout.addRow("Model:", model_layout)
        
        self.target_lang = QComboBox()
        self.target_lang.addItems(["Chinese", "English", "Japanese", "French", "Spanish", "German", "Korean"])
        self.target_lang.setEditable(True)
        self.target_lang.setCurrentText(config.target_lang)
        layout.addRow("Target Language:", self.target_lang)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "🈵 Translation")

    def populate_devices(self):
        self.device_combo.clear()
        self.device_combo.addItem("Auto (Default)", "auto")
        
        try:
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0:
                    name = f"[{i}] {d['name']}"
                    self.device_combo.addItem(name, i) # Store index as data
            
            # Select current
            if config.device_index is not None:
                index = self.device_combo.findData(config.device_index)
                if index >= 0:
                    self.device_combo.setCurrentIndex(index)
        except Exception as e:
            self.device_combo.addItem(f"Error: {e}")

    def save_config(self):
        import configparser
        import os
        
        # Update config object logic would go here, 
        # For now, we write directly to config.ini similarly to settings_window.py
        
        cp = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(__file__), "config.ini")
        cp.read(config_path)
        
        if not cp.has_section("audio"): cp.add_section("audio")
        if not cp.has_section("api"): cp.add_section("api")
        if not cp.has_section("translation"): cp.add_section("translation")
        if not cp.has_section("transcription"): cp.add_section("transcription")
        
        # Audio
        idx = self.device_combo.currentData()
        cp.set("audio", "device_index", str(idx) if idx is not None else "auto")
        cp.set("audio", "sample_rate", str(self.sample_rate.value()))
        cp.set("audio", "silence_threshold", str(self.silence_thresh.value()))
        cp.set("audio", "silence_duration", str(self.silence_dur.value()))
        
        # Transcription
        cp.set("transcription", "whisper_model", self.whisper_model.currentText())
        cp.set("transcription", "device", self.device_type.currentText())
        cp.set("transcription", "compute_type", self.compute_type.currentText())
        cp.set("transcription", "source_language", self.source_language.currentText())
        
        # Translation
        cp.set("api", "api_key", self.api_key.text())
        cp.set("api", "base_url", self.base_url.text())
        cp.set("translation", "model", self.model.currentText())
        cp.set("translation", "target_lang", self.target_lang.currentText())
        
        with open(config_path, 'w') as f:
            cp.write(f)
            
        self.status_label.setText("Saved! Please restart.")

    def on_start(self):
        # 1. Update UI to Loading State
        self.status_label.setText("Initializing Pipeline... (This may take a moment)")
        self.status_label.setStyleSheet("font-size: 18px; color: #fab387;") # Orange for loading
        self.start_btn.setEnabled(False)
        self.start_btn.setText("Loading...")
        
        # 2. Start Worker Thread
        from PyQt6.QtCore import QThread, pyqtSignal
        self.startup_worker = StartupWorker()
        self.startup_worker.finished.connect(self.on_pipeline_ready)
        self.startup_worker.start()

    def on_pipeline_ready(self, _, pipeline):
        # Create Window on Main Thread
        from main import OverlayWindow
        from config import config
        
        if not pipeline:
             self.status_label.setText("Initialization Failed Check Console")
             self.start_btn.setEnabled(True)
             self.start_btn.setText("▶ Launch Translator")
             return

        self.pipeline = pipeline
        self.overlay_window = OverlayWindow(
            display_duration=config.display_duration,
            window_width=config.window_width
        )
        self.overlay_window.show()

        # Connect Signals
        self.pipeline.signals.update_text.connect(self.overlay_window.update_text)
        if hasattr(self.overlay_window, 'stop_requested'):
             self.overlay_window.stop_requested.connect(self.on_stop)

        # Start Pipeline Thread
        self.pipeline.start()

        self.status_label.setText("Running...")
        self.status_label.setStyleSheet("font-size: 18px; color: #a6e3a1;")
        
        self.start_btn.hide()
        self.stop_btn.show()
        
        self.showMinimized()

    def on_stop(self):
        if hasattr(self, 'pipeline') and self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
            
        if hasattr(self, 'overlay_window') and self.overlay_window:
            self.overlay_window.close()
            self.overlay_window = None
            
        self.status_label.setText("Stopped")
        self.stop_btn.hide()
        self.start_btn.show()
        self.start_btn.setEnabled(True)
        self.start_btn.setText("▶ Launch Translator")
        self.showNormal()

class StartupWorker(QThread):
    finished = pyqtSignal(object, object) # window(None), pipeline

    def run(self):
        try:
            from main import Pipeline
            pipeline = Pipeline()
            self.finished.emit(None, pipeline)
        except Exception as e:
            print(f"Startup Error: {e}")
            import traceback
            traceback.print_exc()
            self.finished.emit(None, None)

if __name__ == "__main__":
    def exception_hook(exctype, value, traceback_obj):
        import traceback
        traceback_str = ''.join(traceback.format_tb(traceback_obj))
        error_msg = f"Unhandled Exception: {value}\n\n{traceback_str}"
        print(error_msg)
        from PyQt6.QtWidgets import QMessageBox
        if QApplication.instance():
            QMessageBox.critical(None, "Crash", error_msg)
        else:
            # If no app, just print (already done)
            pass
        sys.exit(1)

    sys.excepthook = exception_hook

    app = QApplication(sys.argv)
    w = Dashboard()
    w.show()
    sys.exit(app.exec())
