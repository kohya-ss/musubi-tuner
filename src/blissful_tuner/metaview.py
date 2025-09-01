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
import math
import subprocess
import argparse
import os
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QTextEdit, QPushButton, QGridLayout, QSizePolicy
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
    parser.add_argument("input_file", help="Input MKV or PNG file")
    parser.add_argument(
        "--columns",
        type=int,
        default=0,
        help="Number of metadata columns (0 = auto based on --max-rows)."
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=12,
        help="When --columns=0, wrap to a new column after this many rows."
    )
    args = parser.parse_args()

    metadata = parse_bt_tags(args.input_file)
    if not metadata:
        print("No bt_ tags found in that file.", file=sys.stderr)
        sys.exit(1)

    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("Blissful Metadata Viewer")
    layout = QGridLayout()
    layout.setHorizontalSpacing(12)
    layout.setVerticalSpacing(6)

    desired_order = [
        "bt_model_type", "bt_task", "bt_prompt", "bt_negative_prompt", "bt_nag_prompt",
        "bt_seeds", "bt_infer_steps", "bt_embedded_cfg_scale", "bt_guidance_scale",
        "bt_cfg_schedule", "bt_fps"
    ]

    # Order keys: desired first (if present), then alphabetical extras
    lower_to_original = {k.lower(): k for k in metadata.keys()}
    ordered_keys = [lower_to_original[k] for k in desired_order if k in lower_to_original]
    extras = sorted(
        [k for k in metadata.keys() if k.lower() not in desired_order],
        key=lambda k: k.lower()
    )
    ordered_keys += extras

    # Decide columns/rows
    n_items = len(ordered_keys)
    if args.columns and args.columns > 0:
        columns = args.columns
        rows_per_col = math.ceil(n_items / columns)
    else:
        rows_per_col = max(1, args.max_rows)
        columns = math.ceil(n_items / rows_per_col)

    # Each "field" occupies 3 grid columns: label, editor, copy
    triplet_width = 3

    clipboard = app.clipboard()

    for i, key in enumerate(ordered_keys):
        r = i % rows_per_col
        c_block = i // rows_per_col
        base_c = c_block * triplet_width

        display_label = key.upper() + ":"
        label = QLabel(display_label)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(label, r, base_c + 0)

        if key.lower() in ["bt_prompt", "bt_negative_prompt", "bt_nag_prompt"]:
            editor = QTextEdit()
            editor.setPlainText(metadata[key])
            fh = editor.fontMetrics().lineSpacing()
            editor.setFixedHeight(fh * 2 + 10)
            # make wide fields not try to expand forever
            editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            editor.setLineWrapMode(QTextEdit.WidgetWidth)
        else:
            editor = QLineEdit(metadata[key])
            editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        editor.setReadOnly(True)
        layout.addWidget(editor, r, base_c + 1)

        copy_btn = QPushButton("Copy")
        copy_btn.clicked.connect(
            lambda _checked, w=editor: clipboard.setText(
                w.toPlainText() if isinstance(w, QTextEdit) else w.text()
            )
        )
        layout.addWidget(copy_btn, r, base_c + 2)

    # Place the Okay! button at the bottom-right of the last column
    ok_button = QPushButton("Okay!")
    ok_button.clicked.connect(window.close)

    # figure out the last logical row/column you used
    last_row = layout.rowCount()
    last_col = layout.columnCount() - 1

    layout.setRowStretch(last_row, 1)

    # now place the button in that bottom-right cell
    layout.addWidget(ok_button, last_row, last_col, alignment=Qt.AlignRight | Qt.AlignBottom)

    window.setLayout(layout)
    # Give it a reasonable default size so multi-column layouts breathe
    window.resize(1280, 720)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
