#!/usr/bin/env python3
"""
Thin wrapper for convert_masks_to_alpha utility.

Converts separate image and mask files into single RGBA PNG files where
the alpha channel contains the mask data. This eliminates filename mismatch
bugs that occur with separate mask directories.

Usage:
    python convert_masks_to_alpha.py --help
"""

from musubi_tuner.convert_masks_to_alpha import main

if __name__ == "__main__":
    main()
