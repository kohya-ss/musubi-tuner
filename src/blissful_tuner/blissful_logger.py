#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 13:24:47 2025

@author: blyss
"""

import logging
from rich.logging import RichHandler


class BlissfulLogger:
    def __init__(self, logging_source: str, log_color: str, do_announce: bool = False):
        self.logging_source = logging_source
        self.log_color = log_color

        # grab (or create) the named logger
        self.logger = logging.getLogger(self.logging_source)
        self.logger.setLevel(logging.DEBUG)

        # DON’T let it propagate to root (which has its own handler)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        # now add one RichHandler
        self.handler = RichHandler(show_time=False, show_level=True, show_path=True, rich_tracebacks=True, markup=True)
        fmt = f"[{self.log_color} bold]%(name)s[/] | %(message)s [dim](%(funcName)s)[/]"
        self.handler.setFormatter(logging.Formatter(fmt))
        self.logger.addHandler(self.handler)
        self.logger.propagate = False

        if do_announce:
            self.logger.info("Set up logging!")

    def set_color(self, new_color):
        self.log_color = new_color
        formatter = logging.Formatter(f"[{self.log_color} bold]%(name)s[/] | %(message)s [dim](%(funcName)s)[/]")
        self.handler.setFormatter(formatter)

    def set_name(self, new_name):
        self.logging_source = "{:<8}".format(new_name)
        self.logger = logging.getLogger(self.logging_source)
        self.logger.setLevel(logging.DEBUG)

        # Remove any existing handlers (just in case)
        if not self.logger.hasHandlers():
            self.logger.addHandler(self.handler)
        else:
            self.logger.handlers.clear()
            self.logger.addHandler(self.handler)

    def info(self, msg):
        self.logger.info(msg, stacklevel=2)

    def debug(self, msg):
        self.logger.debug(msg, stacklevel=2)

    def warning(self, msg, levelmod=0):
        self.logger.warning(msg, stacklevel=2 + levelmod)

    def warn(self, msg):
        self.logger.warning(msg, stacklevel=2)

    def error(self, msg, levelmod=0):
        self.logger.error(msg, stacklevel=2 + levelmod)

    def critical(self, msg):
        self.logger.critical(msg, stacklevel=2)

    def setLevel(self, level):
        self.logger.set_level(level)
