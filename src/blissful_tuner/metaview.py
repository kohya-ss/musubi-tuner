#!/usr/bin/env python3
"""
BT Metadata Viewer

Displays specified bt_ metadata tags from an MKV or PNG file in a simple PySide6 GUI.
Requires:
  - PySide6 ('pip install PySide6' or installed at the system level)
  - Pillow ('pip install Pillow') for PNG metadata extraction
  - mediainfo CLI installed (e.g., 'apt install mediainfo') for MKV files

License: Apache 2.0
"""
import sys
import subprocess
import argparse
import os
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
from PIL import Image  # for reading PNG metadata


def parse_bt_tags(filename: str) -> dict:
    """
    If the file is a PNG, open it with Pillow and extract any metadata keys starting with 'bt_' (case-insensitive)
    from image.info. Otherwise, run mediainfo on the file and extract tags beginning with 'bt_' (case-insensitive).
    Returns a dict mapping tag -> value, preserving whatever case the key was originally stored as.
    """
    tags = {}
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext == ".png":
        # --- PNG handling via Pillow ---
        try:
            img = Image.open(filename)
        except Exception as e:
            print(f"Error opening PNG: {e}", file=sys.stderr)
            sys.exit(1)

        # Pillow’s Image.info is a dict of any text chunks (tEXt/iTXt) and some other metadata.
        # We pick out keys where key.lower().startswith("bt_").
        for key, value in img.info.items():
            if key.lower().startswith("bt_"):
                tags[key] = value
        return tags

    else:
        # --- MKV (or other) handling via mediainfo ---
        try:
            output = subprocess.check_output(["mediainfo", filename], text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running mediainfo: {e}", file=sys.stderr)
            sys.exit(1)

        for line in output.splitlines():
            # mediainfo might show lines like "bt_prompt                        : some text"
            # so do a case-insensitive check:
            if ":" in line:
                left, right = line.split(":", 1)
                if left.strip().lower().startswith("bt_"):
                    key = left.strip()
                    value = right.strip()
                    tags[key] = value
        return tags


def main():
    parser = argparse.ArgumentParser(
        description="Display bt_ metadata tags from an MKV or PNG in a GUI"
    )
    parser.add_argument(
        "input_file",
        help="Input MKV or PNG file"
    )
    args = parser.parse_args()

    metadata = parse_bt_tags(args.input_file)
    if not metadata:
        print("No bt_ tags found in that file.", file=sys.stderr)
        sys.exit(1)

    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("BT Metadata Viewer")
    layout = QGridLayout()

    desired_order = [
        "bt_model_type",
        "bt_task",
        "bt_prompt",
        "bt_negative_prompt",
        "bt_seeds",
        "bt_infer_steps",
        "bt_embedded_cfg_scale",
        "bt_guidance_scale",
        "bt_cfg_schedule",
        "bt_fps"
    ]

    # Order the keys: first in desired_order (if present), then any others sorted alphabetically
    # We do a case-insensitive match to see if the stored key lowercased is in desired_order
    ordered_keys = []
    lower_to_original = {k.lower(): k for k in metadata.keys()}
    for want in desired_order:
        if want in lower_to_original:
            ordered_keys.append(lower_to_original[want])

    # Any metadata keys that weren't in desired_order, sorted by lowercase name
    extras = sorted(
        [orig_key for orig_key in metadata.keys() if orig_key.lower() not in desired_order],
        key=lambda k: k.lower()
    )
    ordered_keys += extras

    clipboard = app.clipboard()
    row = 0

    # Create UI rows for each tag
    for key in ordered_keys:
        display_label = key.upper() + ":"  # e.g. "bt_prompt" → "BT_PROMPT:"
        label = QLabel(display_label)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(label, row, 0)

        # Use QTextEdit for long fields like prompt; otherwise QLineEdit
        if key.lower() in ["bt_prompt", "bt_negative_prompt"]:
            editor = QTextEdit()
            editor.setPlainText(metadata[key])
            fh = editor.fontMetrics().lineSpacing()
            editor.setFixedHeight(fh * 4 + 10)
        else:
            editor = QLineEdit(metadata[key])

        editor.setReadOnly(True)
        layout.addWidget(editor, row, 1)

        copy_btn = QPushButton("Copy")
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
