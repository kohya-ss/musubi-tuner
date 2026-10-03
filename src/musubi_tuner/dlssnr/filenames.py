"""Portable single-component names for NR run directories and sample outputs."""

from pathlib import PureWindowsPath


def validate_filename(value: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value in (".", "..")
        or value.endswith((".", " "))
        or any(ord(char) < 32 or char in '/\\:<>"|?*' for char in value)
        or PureWindowsPath(value).is_reserved()
    ):
        raise ValueError(f"{field} must be a nonempty, filename-safe name on Windows and POSIX")
