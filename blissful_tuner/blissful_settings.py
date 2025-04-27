#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 19:08:55 2025

@author: blyss
"""
import os
import json


class SingletonMeta(type):
    """
    The SingletonMeta class is useful for creating objects that persist as a single instance across the whole program. Basically a global class.
    """
    _instances = {}

    def __call__(cls, *Parameters, **kwParameters):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*Parameters, **kwParameters)
        return cls._instances[cls]


class BlissfulSettings(metaclass=SingletonMeta):
    def __init__(self):
        """
        Loads the settings from a 'settings.json' file, creating it with default settings if it doesn't exist.

        This method attempts to read the program's settings from a JSON file. If the file does not exist,
        it creates a new file with default settings. This ensures that the program can start with a known
        set of configurations and modify them as needed.

        This class is a SingletonMeta so even if we reinstantiate the class, this only happens the first time
        """
        # These are globals that do not persist
        self.generating = 0
        self.last_preview_file = ""

        default_settings = {
            "prompt": "a cat walks on the grass, realistic style",
            "resolution_x": 960,
            "resolution_y": 544,
            "fps": 24,
            "embedded_guidance": 6.0,
            "flow_shift": 7.0,
            "infer_steps": 50,
            "seed": 42,
            "video_length": 129,
            "attention": "sage",
            "blocks_to_swap": 0,
            "hidden_state_skip_layer": 2,
            "apply_final_norm": False,
            "reproduce": False,
            "fp8": True,
            "fp8_fast": False,
            "do_compile": False,
            "transformer_path": "",
            "text_encoder_1_path": "",
            "text_encoder_2_path": "",
            "vae_path": "",
            "lora_path": "",
        }

        if not os.path.exists("./settings.json"):
            with open("./settings.json", "w", encoding="utf-8") as file:
                json.dump(default_settings, file, indent=4)
            print("No existing settings found. Created default settings file.")

        with open("./settings.json", "r", encoding="utf-8") as file:
            data = json.load(file)

        for key, default_value in default_settings.items():
            setattr(self, key, data.get(key, default_value))

    def save_to_file(self):
        """
        Saves the current settings to a JSON file named 'settings.json'.
        """
        settings = {
            "prompt": self.prompt,
            "resolution_x": self.resolution_x,
            "resolution_y": self.resolution_y,
            "fps": self.fps,
            "embedded_guidance": self.embedded_guidance,
            "flow_shift": self.flow_shift,
            "infer_steps": self.infer_steps,
            "seed": self.seed,
            "video_length": self.video_length,
            "attention": self.attention,
            "blocks_to_swap": self.blocks_to_swap,
            "hidden_state_skip_layer": self.hidden_state_skip_layer,
            "apply_final_norm": self.apply_final_norm,
            "reproduce": self.reproduce,
            "fp8": self.fp8,
            "fp8_fast": self.fp8_fast,
            "do_compile": self.do_compile,
            "transformer_path": self.transformer_path,
            "text_encoder_1_path": self.text_encoder_1_path,
            "text_encoder_2_path": self.text_encoder_2_path,
            "vae_path": self.vae_path,
            "lora_path": self.lora_path,
        }

        with open("./settings.json", "w", encoding="utf-8") as file:
            json.dump(settings, file, indent=4)

    def update(self, option, value, label_target=None, label_value=None):
        """Method for updating various settings called via QT connection and may update an associated label/value"""
        setattr(self, option, value)
        if label_target is not None and label_value is not None:
            label_target.setText(str(label_value))
