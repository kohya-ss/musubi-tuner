#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 13:59:54 2025

@author: blyss
"""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QDial, QLabel, QLineEdit, QDialog, QPushButton, QFileDialog, QFormLayout, QCheckBox, QTextEdit
from PySide6.QtCore import Qt, Signal, QEvent, QLineF
from PySide6.QtGui import QPainter, QMouseEvent, QIntValidator
from blissful_tuner.blissful_settings import BlissfulSettings


class PromptWidget(QWidget):
    def __init__(self, global_settings):
        super().__init__()
        self.global_settings = global_settings

        main_layout = QVBoxLayout(self)

        prompt_label = QLabel("Prompt:")
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setText(self.global_settings.prompt)
        self.prompt_edit.textChanged.connect(self.on_prompt_text_changed)

        main_layout.addWidget(prompt_label)
        main_layout.addWidget(self.prompt_edit)

    def on_prompt_text_changed(self):
        text = self.prompt_edit.toPlainText()
        self.global_settings.update("prompt", text)


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.global_settings = BlissfulSettings()
        self.setWindowTitle("Settings")
        self.resize(500, 300)
        main_layout = QVBoxLayout(self)

        # QFormLayout with right-aligned labels.
        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignRight)
        self.path_edits = {}

        settings = ["Transformer", "Text Encoder 1", "Text Encoder 2", "VAE", "LoRAs"]

        self.attr_mapping = {
            "Transformer": "transformer_path",
            "Text Encoder 1": "text_encoder_1_path",
            "Text Encoder 2": "text_encoder_2_path",
            "VAE": "vae_path",
            "LoRAs": "lora_path"
        }

        for setting in settings:
            container = QWidget()
            container_layout = QHBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)

            line_edit = QLineEdit()
            line_edit.setMinimumWidth(300)
            current_value = getattr(self.global_settings, self.attr_mapping[setting], "")
            line_edit.setText(current_value)

            browse_button = QPushButton("Browse")
            browse_button.clicked.connect(lambda checked, le=line_edit, s=setting: self.open_file_or_folder(le, s))

            container_layout.addWidget(line_edit)
            container_layout.addWidget(browse_button)

            self.path_edits[setting] = line_edit
            form_layout.addRow(QLabel(f"{setting}:"), container)

        main_layout.addLayout(form_layout)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        main_layout.addLayout(button_layout)

    def open_file_or_folder(self, line_edit, setting):
        """
        Opens a file dialog for settings other than LoRAs, and a folder dialog for LoRAs.
        """
        if setting == "LoRAs":
            folder = QFileDialog.getExistingDirectory(self, "Select Folder")
            if folder:
                line_edit.setText(folder)
        else:
            file, _ = QFileDialog.getOpenFileName(self, "Select File")
            if file:
                line_edit.setText(file)

    def accept(self):
        for key, value in self.path_edits.items():
            setattr(self.global_settings, self.attr_mapping[key], value.text())
        super().accept()


class SeedWidget(QWidget):
    def __init__(self, global_settings):
        super().__init__()
        self.global_settings = global_settings

        main_layout = QVBoxLayout(self)
        self.line_edit = QLineEdit(str(self.global_settings.seed))
        self.line_edit.setFixedWidth(80)
        self.line_edit.setValidator(QIntValidator(-999999999, 999999999, self))
        self.line_edit.textChanged.connect(lambda value: self.global_settings.update("seed", value))
        main_layout.addWidget(QLabel("Seed"), alignment=Qt.AlignCenter)
        main_layout.addWidget(self.line_edit, alignment=Qt.AlignCenter)

        checkbox_layout = QHBoxLayout()
        self.checkbox = QCheckBox("Random")
        self.checkbox.toggled.connect(self.on_checkbox_toggled)
        checkbox_layout.addWidget(self.checkbox, alignment=Qt.AlignCenter)

        main_layout.addLayout(checkbox_layout)

    def on_checkbox_toggled(self, checked):
        self.line_edit.setText("-1")
        self.line_edit.setDisabled(checked)
        self.global_settings.seed = -1


class BlissfulDial(QDial):
    """Wraps the QDial so we can write the value to it's center like a boss(or like someone who cares about UI design)"""

    def paintEvent(self, event):
        super().paintEvent(event)
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        value = self.value()
        qp.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(value))


class ValueDial(QWidget):
    """A QDial that displays values on it's notches, ported from Qt5 version here https://stackoverflow.com/questions/63698714/how-to-show-markings-on-qdial-in-pyqt5-python"""
    _dialProperties = ('minimum', 'maximum', 'value', 'singleStep', 'pageStep',
                       'notchesVisible', 'tracking', 'wrapping',
                       'invertedAppearance', 'invertedControls', 'orientation')
    _inPadding = 3
    _outPadding = 2
    valueChanged = Signal(int)

    def __init__(self, *args, **kwargs):
        # Remove properties used as keyword arguments for the dial.
        dialArgs = {k: v for k, v in kwargs.items() if k in self._dialProperties}
        for k in dialArgs.keys():
            kwargs.pop(k)
        super().__init__(*args, **kwargs)
        layout = QVBoxLayout(self)
        self.dial = BlissfulDial(self, **dialArgs)
        layout.addWidget(self.dial)
        self.dial.valueChanged.connect(self.valueChanged)
        # Make the dial the focus proxy (so that it captures focus *and* key events)
        self.setFocusProxy(self.dial)

        # Simple "monkey patching" to access dial functions.
        self.value = self.dial.value
        self.setValue = self.dial.setValue
        self.minimum = self.dial.minimum
        self.maximum = self.dial.maximum
        self.wrapping = self.dial.wrapping
        self.notchesVisible = self.dial.notchesVisible
        self.setNotchesVisible = self.dial.setNotchesVisible
        self.setNotchTarget = self.dial.setNotchTarget
        self.notchSize = self.dial.notchSize
        self.invertedAppearance = self.dial.invertedAppearance
        self.setInvertedAppearance = self.dial.setInvertedAppearance

        self.updateSize()

    def inPadding(self):
        return self._inPadding

    def setInPadding(self, padding):
        self._inPadding = max(0, padding)
        self.updateSize()

    def outPadding(self):
        return self._outPadding

    def setOutPadding(self, padding):
        self._outPadding = max(0, padding)
        self.updateSize()

    def setMinimum(self, minimum):
        self.dial.setMinimum(minimum)
        self.updateSize()

    def setMaximum(self, maximum):
        self.dial.setMaximum(maximum)
        self.updateSize()

    def setWrapping(self, wrapping):
        self.dial.setWrapping(wrapping)
        self.updateSize()

    def updateSize(self):
        # Update margins so that the value strings always have enough space.
        fm = self.fontMetrics()
        minWidth = max(fm.horizontalAdvance(str(v)) for v in range(self.minimum(), self.maximum() + 1))
        self.offset = max(minWidth, fm.height()) / 2
        margin = self.offset + self._inPadding + self._outPadding
        self.layout().setContentsMargins(margin, margin, margin, margin)

    def translateMouseEvent(self, event):
        # Translate mouse events to the dial.
        return QMouseEvent(
            event.type(),
            self.dial.mapFrom(self, event.pos()),
            event.button(),
            event.buttons(),
            event.modifiers()
        )

    def changeEvent(self, event):
        if event.type() == QEvent.Type.FontChange:
            self.updateSize()

    def mousePressEvent(self, event):
        self.dial.mousePressEvent(self.translateMouseEvent(event))

    def mouseMoveEvent(self, event):
        self.dial.mouseMoveEvent(self.translateMouseEvent(event))

    def mouseReleaseEvent(self, event):
        self.dial.mouseReleaseEvent(self.translateMouseEvent(event))

    def paintEvent(self, event):
        radius = min(self.width(), self.height()) / 2
        radius -= (self.offset / 2 + self._outPadding)
        invert = -1 if self.invertedAppearance() else 1
        if self.wrapping():
            angleRange = 360
            startAngle = 270
            rangeOffset = 0
        else:
            angleRange = 300
            startAngle = 240 if invert > 0 else 300
            rangeOffset = 1
        fm = self.fontMetrics()

        # Reference line for positioning the text.
        reference = QLineF.fromPolar(radius, 0).translated(self.rect().center())
        fullRange = self.maximum() - self.minimum()
        textRect = self.rect()

        qp = QPainter(self)
        qp.setRenderHints(QPainter.RenderHint.Antialiasing)
        label_interval = 4  # Print every Nth numberal label
        for p in range(0, fullRange + rangeOffset, self.notchSize() * label_interval):
            value = self.minimum() + p
            if invert < 0:
                value -= 1
                if value < self.minimum():
                    continue
            angle = p / fullRange * angleRange * invert
            reference.setAngle(startAngle - angle)
            textRect.setSize(fm.size(Qt.TextFlag.TextSingleLine, str(value)))
            textRect.moveCenter(reference.p2().toPoint())
            qp.drawText(textRect, Qt.AlignmentFlag.AlignCenter, str(value))


class ResolutionWidget(QWidget):
    """Custom widget for specifying resolution and framerate with validation for the former"""

    def __init__(self, global_settings):
        super().__init__()
        self.global_settings = global_settings
        self.initUI()

    def initUI(self):
        layout = QHBoxLayout(self)

        self.width_input = QLineEdit()
        self.width_input.setText(str(self.global_settings.resolution_x))
        self.width_input.setFixedWidth(60)
        self.width_input.setValidator(QIntValidator(1, 9999, self))
        self.check_divisible(self.width_input)
        self.width_input.textChanged.connect(lambda: self.check_divisible(self.width_input))
        self.width_input.textChanged.connect(lambda value: self.global_settings.update("resolution_x", value))

        x_label = QLabel("x")

        self.height_input = QLineEdit()
        self.height_input.setText(str(self.global_settings.resolution_y))
        self.height_input.setFixedWidth(60)
        self.height_input.setValidator(QIntValidator(1, 9999, self))
        self.check_divisible(self.height_input)
        self.height_input.textChanged.connect(lambda: self.check_divisible(self.height_input))
        self.height_input.textChanged.connect(lambda value: self.global_settings.update("resolution_y", value))

        at_label = QLabel("@")

        self.fps_input = QLineEdit()
        self.fps_input.setText(str(self.global_settings.fps))
        self.fps_input.setFixedWidth(25)
        self.fps_input.setValidator(QIntValidator(1, 200, self))
        self.fps_input.textChanged.connect(lambda value: self.global_settings.update("fps", value))

        fps_label = QLabel("fps")

        layout.addWidget(self.width_input, alignment=Qt.AlignLeft)
        layout.addWidget(x_label, alignment=Qt.AlignLeft)
        layout.addSpacing(-30)
        layout.addWidget(self.height_input, alignment=Qt.AlignLeft)
        layout.addWidget(at_label, alignment=Qt.AlignLeft)
        layout.addSpacing(-30)
        layout.addWidget(self.fps_input, alignment=Qt.AlignLeft)
        layout.addSpacing(-20)
        layout.addWidget(fps_label, alignment=Qt.AlignLeft)

    def check_divisible(self, line_edit: QLineEdit):
        """
        Check if the number in the QLineEdit is divisible by 8.
        If yes, set the background color to light green.
        If not, set it to light red (light coral).
        """
        text = line_edit.text().strip()
        if text:
            try:
                value = int(text)
                if value % 8 == 0:
                    # Divisible by 8 -> light green background
                    line_edit.setStyleSheet("background-color: darkgreen;")
                else:
                    # Not divisible by 8 -> light red background
                    line_edit.setStyleSheet("background-color: lightcoral;")
            except ValueError:
                # If conversion fails, mark as invalid (red background)
                line_edit.setStyleSheet("background-color: lightcoral;")
        else:
            # Empty input, remove background color
            line_edit.setStyleSheet("")
