from PyQt6.QtWidgets import (QApplication, QWidget, QTextEdit, QVBoxLayout, QGraphicsDropShadowEffect, 
                             QSizeGrip, QHBoxLayout, QScrollArea, QLabel, QFrame)
from PyQt6.QtCore import Qt, QPoint, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette

from ctypes import c_void_p
import time

# macOS: Make window visible on all desktops (Spaces)
try:
    from AppKit import NSWindowCollectionBehaviorCanJoinAllSpaces, NSWindowCollectionBehaviorStationary
    import objc
    HAS_APPKIT = True
except ImportError:
    HAS_APPKIT = False

class LogItem(QFrame):
    """A widget representing a single chunk of transcription/translation"""
    def __init__(self, chunk_id, timestamp, original_text, translated_text="", speaker=""):
        super().__init__()
        self.chunk_id = chunk_id
        
        # Style
        self.setStyleSheet("background-color: transparent;")
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 15) # Bottom margin
        self.layout.setSpacing(2)
        self.setLayout(self.layout)
        
        # Speaker label (if available)
        if speaker:
            self.speaker_label = QLabel(f"🎤 {speaker}")
            self.speaker_label.setWordWrap(True)
            self.speaker_label.setStyleSheet("color: #89b4fa; font-family: Arial; font-size: 12px; font-weight: bold;")
            self.layout.addWidget(self.speaker_label)
        else:
            self.speaker_label = None
        
        # Original Text Label
        self.original_label = QLabel(f"[{timestamp}] {original_text}")
        self.original_label.setWordWrap(True)
        self.original_label.setStyleSheet("color: #aaaaaa; font-family: Arial; font-size: 14px;")
        self.layout.addWidget(self.original_label)
        
        # Translated Text Label
        self.translated_label = QLabel(translated_text if translated_text else "...")
        self.translated_label.setWordWrap(True)
        self.translated_label.setStyleSheet("color: #ffffff; font-family: Arial; font-size: 20px; font-weight: bold;")
        self.layout.addWidget(self.translated_label)
        
    def update_translated(self, text):
        self.translated_label.setText(text)

    def update_original(self, text):
        self.original_label.setText(f"[{time.strftime('%H:%M:%S')}] {text}")
    
    def update_speaker(self, speaker):
        """Update speaker label with visual feedback"""
        if speaker and not self.speaker_label:
            # Create speaker label if it doesn't exist
            self.speaker_label = QLabel(f"🎤 {speaker}")
            self.speaker_label.setWordWrap(True)
            self.speaker_label.setStyleSheet("color: #89b4fa; font-family: Arial; font-size: 12px; font-weight: bold;")
            self.layout.insertWidget(0, self.speaker_label)  # Insert at top
        elif speaker and self.speaker_label:
            # Update existing speaker label with color change animation
            self.speaker_label.setText(f"🎤 {speaker}")
            # Flash the speaker label to show it updated
            self.speaker_label.setStyleSheet("color: #a6e3a1; font-family: Arial; font-size: 12px; font-weight: bold;")  # Green flash
            QTimer.singleShot(500, lambda: self.speaker_label.setStyleSheet("color: #89b4fa; font-family: Arial; font-size: 12px; font-weight: bold;"))  # Back to blue

class OverlayWindow(QWidget):
    def __init__(self, display_duration=None, window_width=400, window_height=None):
        super().__init__()
        # display_duration is not really used in log mode, but kept for compatibility
        self.window_width = window_width
        
        # Default height to full screen height if not specified
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        self.window_height = window_height if window_height else screen_geometry.height()
        
        self.initUI()
        self.oldPos = self.pos()

    def showEvent(self, event):
        """Called when window is shown - set all-spaces behavior here"""
        super().showEvent(event)
        if HAS_APPKIT:
            self._set_all_spaces()
    
    def _set_all_spaces(self):
        """Make window appear on all macOS Spaces/Desktops"""
        try:
            # Get the native NSWindow from Qt's winId
            win_id = int(self.winId())
            ns_view = objc.objc_object(c_void_p=c_void_p(win_id))
            ns_window = ns_view.window()
            ns_window.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorStationary
            )
            print("Window set to appear on all Spaces")
        except Exception as e:
            print(f"Could not set all-spaces behavior: {e}")

    def initUI(self):
        # Window flags for transparency and staying on top
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.WindowDoesNotAcceptFocus
        )
class ResizeHandle(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_window = parent
        self.setText("◢")
        self.setStyleSheet("color: rgba(255, 255, 255, 100); font-size: 16px;")
        self.setFixedSize(20, 20)
        self.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
        self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        
        self.startPos = None
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.startPos = event.globalPosition().toPoint()
            event.accept()
            
    def mouseMoveEvent(self, event):
        if self.startPos:
            delta = event.globalPosition().toPoint() - self.startPos
            new_width = max(self.parent_window.minimumWidth(), self.parent_window.width() + delta.x())
            new_height = max(self.parent_window.minimumHeight(), self.parent_window.height() + delta.y())
            
            self.parent_window.resize(new_width, new_height)
            self.startPos = event.globalPosition().toPoint()
            event.accept()
            
    def mouseReleaseEvent(self, event):
        self.startPos = None

class OverlayWindow(QWidget):
    stop_requested = pyqtSignal()

    def __init__(self, display_duration=None, window_width=400, window_height=None):
        super().__init__()
        # display_duration is not really used in log mode, but kept for compatibility
        self.window_width = window_width
        
        # Default height to full screen height if not specified
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        self.window_height = window_height if window_height else screen_geometry.height()
        
        self.initUI()
        self.oldPos = self.pos()

    def showEvent(self, event):
        """Called when window is shown - set all-spaces behavior here"""
        super().showEvent(event)
        if HAS_APPKIT:
            self._set_all_spaces()
    
    def _set_all_spaces(self):
        """Make window appear on all macOS Spaces/Desktops"""
        try:
            # Get the native NSWindow from Qt's winId
            win_id = int(self.winId())
            ns_view = objc.objc_object(c_void_p=c_void_p(win_id))
            ns_window = ns_view.window()
            ns_window.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorStationary
            )
            print("Window set to appear on all Spaces")
        except Exception as e:
            print(f"Could not set all-spaces behavior: {e}")

    def initUI(self):
        # Window flags for transparency and staying on top
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        
        # Layout
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(layout)
        
        # SCROLL AREA
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        # Transparent scroll area
        self.scroll_area.setStyleSheet("""
            QScrollArea { background: transparent; }
            QScrollBar:vertical { width: 0px; }
        """)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Container for LogItems
        self.container = QFrame()
        self.container.setStyleSheet("background-color: rgba(0, 0, 0, 150); border-radius: 10px;")
        self.container_layout = QVBoxLayout()
        self.container_layout.setContentsMargins(10, 10, 10, 10)
        # Allocate alignment to top so items stack from top
        self.container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.container.setLayout(self.container_layout)
        
        self.scroll_area.setWidget(self.container)
        layout.addWidget(self.scroll_area)
        
        # Bottom Control Bar (Resize Grip + Save Button)
        grip_layout = QHBoxLayout()
        
        # Save Button
        from PyQt6.QtWidgets import QPushButton, QStyle 
        self.save_btn = QPushButton("💾 Save")
        self.save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_btn.setFixedWidth(80)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 50);
                color: white;
                border-radius: 5px;
                padding: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 100);
            }
        """)
        self.save_btn.clicked.connect(self._save_transcript)
        
        grip_layout.addWidget(self.save_btn)
        
        # Stop Button
        self.stop_btn = QPushButton("⏹")
        self.stop_btn.setToolTip("Stop Translator")
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setFixedSize(30, 30)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(243, 139, 168, 150);
                color: white;
                border-radius: 15px;
                border: none;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: rgba(243, 139, 168, 200);
            }
        """)
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        grip_layout.addWidget(self.stop_btn)
        
        grip_layout.addStretch()
        
        # Visual Grip Indicator
        self.grip_label = ResizeHandle(self)
        grip_layout.addWidget(self.grip_label)
        
        layout.addLayout(grip_layout)
        
        # Set initial window size
        self.resize(self.window_width, self.window_height)
        
        # Position: Right side of screen, full height
        screen = QApplication.primaryScreen().availableGeometry()
        x = screen.x() + screen.width() - self.window_width - 20 # 20px padding from right
        y = screen.y()
        self.move(x, y)
        
        # Data storage: list of (chunk_id, widget) inclusive
        self.items = [] # Sorted by chunk_id
        
        # History for saving (list of dicts)
        self.transcript_data = {} # chunk_id -> {timestamp, original, translated, speaker}
        
        # State
        self.is_moving = False
        
        # Enable mouse tracking for cursor update without click
        self.setMouseTracking(True)

    def update_text(self, chunk_id, original_text, translated_text, speaker=""):
        """Append new text or update existing text"""
        print(f"[Overlay] Received update for #{chunk_id}: {original_text} -> {translated_text} (Speaker: {speaker})")
        
        # Update data store
        if chunk_id not in self.transcript_data:
            self.transcript_data[chunk_id] = {
                'timestamp': time.strftime("%H:%M:%S"),
                'original': original_text,
                'translated': translated_text,
                'speaker': speaker
            }
        else:
            if original_text:
                self.transcript_data[chunk_id]['original'] = original_text
            if translated_text:
                self.transcript_data[chunk_id]['translated'] = translated_text
            if speaker:
                self.transcript_data[chunk_id]['speaker'] = speaker
        
        # Check if widget exists
        existing_widget = None
        for cid, widget in self.items:
            if cid == chunk_id:
                existing_widget = widget
                break
        
        if existing_widget:
            # Update existing
            if original_text:
                existing_widget.update_original(original_text)
            
            if translated_text:
                existing_widget.update_translated(translated_text)
            
            if speaker:
                existing_widget.update_speaker(speaker)
                
            print(f"[Overlay] Updated existing widget #{chunk_id}")
        else:
            # Insert new widget in order
            timestamp = self.transcript_data[chunk_id]['timestamp']
            new_widget = LogItem(chunk_id, timestamp, original_text, translated_text, speaker)
            
            # Find insertion point
            insert_idx = len(self.items)
            for i, (cid, w) in enumerate(self.items):
                if cid > chunk_id:
                    insert_idx = i
                    break
            
            self.items.insert(insert_idx, (chunk_id, new_widget))
            self.container_layout.insertWidget(insert_idx, new_widget)
            print(f"[Overlay] Inserted new widget #{chunk_id} at index {insert_idx}")
            
            # Scroll to bottom
            QTimer.singleShot(10, self._scroll_to_bottom)
    
    def update_speaker_only(self, chunk_id, speaker):
        """Update only speaker information for a chunk (async diarization result)"""
        print(f"[Overlay] Async speaker update for #{chunk_id}: {speaker}")
        
        # Update data store
        if chunk_id in self.transcript_data:
            self.transcript_data[chunk_id]['speaker'] = speaker
        else:
            # Chunk doesn't exist yet, create placeholder
            self.transcript_data[chunk_id] = {
                'timestamp': time.strftime("%H:%M:%S"),
                'original': '',
                'translated': '',
                'speaker': speaker
            }
        
        # Update widget if it exists
        for cid, widget in self.items:
            if cid == chunk_id:
                widget.update_speaker(speaker)
                print(f"[Overlay] Updated speaker for widget #{chunk_id}")
                break

    def _scroll_to_bottom(self):
        sb = self.scroll_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _save_transcript(self):
        """Save history to file"""
        import os
        if not self.transcript_data:
            print("[Overlay] Nothing to save.")
            return

        os.makedirs("transcripts", exist_ok=True)
        filename = f"transcripts/transcript_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        
        # Sort by chunk_id
        sorted_ids = sorted(self.transcript_data.keys())
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Transcript saved at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n\n")
                for cid in sorted_ids:
                    data = self.transcript_data[cid]
                    speaker_info = f" ({data.get('speaker', '')})" if data.get('speaker') else ""
                    f.write(f"[{data['timestamp']}]{speaker_info} (ID: {cid})\nOriginal: {data['original']}\nTranslation: {data['translated']}\n{'-'*30}\n")
            
            print(f"[Overlay] Saved to {filename}")
            # Visual feedback on button
            original_text = self.save_btn.text()
            self.save_btn.setText("Saved!")
            QTimer.singleShot(2000, lambda: self.save_btn.setText(original_text))
            
        except Exception as e:
            print(f"[Overlay] Error saving transcript: {e}")



    # Window Moving Logic (Resize is handled by ResizeHandle widget)
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_moving = True
            self.oldPos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        # Update cursor shape based on position (reset to arrow)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        
        # Handle dragging
        if self.is_moving:
            delta = event.globalPosition().toPoint() - self.oldPos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.oldPos = event.globalPosition().toPoint()
            
    def mouseReleaseEvent(self, event):
        self.is_moving = False
        self.setCursor(Qt.CursorShape.ArrowCursor)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = OverlayWindow()
    window.show()
    # Test update with speaker
    window.update_text(1, "Hello world", "", "Speaker 1")
    QTimer.singleShot(1000, lambda: window.update_text(1, "Hello world", "你好，世界", "Speaker 1"))
    window.update_text(2, "Sequence test", "", "Speaker 2")
    sys.exit(app.exec())
