#!/usr/bin/env python3
"""
BT Metadata Viewer

Displays specified BT_ metadata tags from an MKV file in a simple PySide6 GUI.
Requires:
  - PySide6 'pip install PySide6' or installed at the system level
  - mediainfo CLI installed (e.g., apt install mediainfo)
License: Apache 2.0
"""
import sys
import subprocess
import argparse
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QTextEdit,
    QPushButton,
    QGridLayout
)
from PySide6.QtCore import Qt


def parse_bt_tags(filename: str) -> dict:
    """
    Run mediainfo on the file and extract all tags beginning with BT_.
    Returns a dict of tag -> value.
    """
    try:
        output = subprocess.check_output(["mediainfo", filename], text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running mediainfo: {e}", file=sys.stderr)
        sys.exit(1)

    tags = {}
    for line in output.splitlines():
        if line.startswith("BT_") and ":" in line:
            key, value = line.split(":", 1)
            tags[key.strip()] = value.strip()
    return tags


def main():
    parser = argparse.ArgumentParser(
        description="Display BT_ metadata tags from an MKV in a GUI"
    )
    parser.add_argument(
        "input_file",
        help="Input MKV file"
    )
    args = parser.parse_args()

    metadata = parse_bt_tags(args.input_file)

    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("BT Metadata Viewer")
    layout = QGridLayout()

    # Define the primary tags in the desired display order
    desired_order = [
        "BT_MODEL_TYPE",
        "BT_TASK",
        "BT_PROMPT",
        "BT_NEGATIVE_PROMPT",
        "BT_SEEDS",
        "BT_INFER_STEPS",
        "BT_EMBEDDED_CFG_SCALE",
        "BT_GUIDANCE_SCALE",
        "BT_CFG_SCHEDULE",
        "BT_FPS"
    ]

    # Order the keys: first in desired_order (if present), then any others sorted alphabetically
    ordered_keys = [k for k in desired_order if k in metadata]
    other_keys = sorted(k for k in metadata if k not in desired_order)
    ordered_keys += other_keys

    clipboard = app.clipboard()
    row = 0

    # Create UI rows for each tag
    for key in ordered_keys:
        label = QLabel(key + ":")
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(label, row, 0)

        if key == "BT_PROMPT":
            editor = QTextEdit()
            editor.setPlainText(metadata[key])
            # approximate height for 3 lines
            fh = editor.fontMetrics().lineSpacing()
            editor.setFixedHeight(fh * 3 + 10)
        else:
            editor = QLineEdit(metadata[key])

        editor.setReadOnly(True)
        layout.addWidget(editor, row, 1)

        copy_btn = QPushButton("Copy")
        # closure captures the editor widget
        copy_btn.clicked.connect(
            lambda _checked, w=editor: clipboard.setText(
                w.toPlainText() if isinstance(w, QTextEdit) else w.text()
            )
        )
        layout.addWidget(copy_btn, row, 2)
        row += 1

    # Add the Okay! button at bottom-right
    ok_button = QPushButton("Okay!")
    ok_button.clicked.connect(window.close)
    layout.addWidget(ok_button, row, 2)

    window.setLayout(layout)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
