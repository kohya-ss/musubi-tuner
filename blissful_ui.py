#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blissful Tuner is a frontend for Musubi Tuner using Qt
Created on Mon Mar 10 16:47:29 2025

@author: blyss
"""
import subprocess
import sys
import os
import shutil
import threading
import random
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QSlider, QLabel, QCheckBox, QGroupBox, QSizePolicy, QPushButton
from PySide6.QtWidgets import QComboBox
from PySide6.QtCore import Qt, QUrl, QTimer
from PySide6.QtGui import QIntValidator
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from blissful_tuner.blissful_settings import BlissfulSettings
from blissful_tuner.widgets import ResolutionWidget, ValueDial, SettingsDialog, SeedWidget, PromptWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.global_settings = BlissfulSettings()
        self.setWindowTitle("Blissful Tuner")
        self.resize(800, 600)

        # Set up the central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # --- Prompt Text Box ---
        self.prompt_edit = PromptWidget(self.global_settings)
        main_layout.addWidget(self.prompt_edit)

        first_row_layout = QHBoxLayout()
        # --- Resolution input w/ validation ---
        first_row_layout.setAlignment(Qt.AlignLeft)
        resolution_label = QLabel("Resolution:")
        resolution_widget = ResolutionWidget(self.global_settings)
        resolution_widget.setFixedWidth(250)
        first_row_layout.addWidget(resolution_label, alignment=Qt.AlignLeft)
        first_row_layout.addWidget(resolution_widget, alignment=Qt.AlignLeft)
        first_row_layout.addSpacing(20)

        # --- Video Length Slider ---
        video_length_tip = "Video length: How many frames in length the generated video should be"
        video_length_label = QLabel("Video Length:")
        video_length_label.setToolTip(video_length_tip)
        self.video_length_value_label = QLabel(f"{self.global_settings.video_length} frame(s)")
        self.video_length_value_label.setToolTip(video_length_tip)
        self.video_length_slider = QSlider(Qt.Horizontal)
        self.video_length_slider.setToolTip(video_length_tip)
        self.video_length_slider.setMinimum(1)
        self.video_length_slider.setMaximum(200)
        self.video_length_slider.setValue(self.global_settings.video_length)
        self.video_length_slider.setMinimumWidth(200)
        self.video_length_slider.valueChanged.connect(lambda value: self.global_settings.update("video_length", value, self.video_length_value_label, f"{value} frame(s)"))
        first_row_layout.addWidget(video_length_label)
        first_row_layout.addWidget(self.video_length_slider)
        first_row_layout.addWidget(self.video_length_value_label)

        main_layout.addLayout(first_row_layout)

        second_row_layout = QHBoxLayout()
        second_row_layout.setAlignment(Qt.AlignLeft)

        slider_group = QGroupBox()
        slider_layout = QVBoxLayout()

        # --- Embedded Guidance Slider ---
        guidance_tip = "Embedded guidance scale: How tightly the model adheres to your prompt, default is 6.0"
        embedded_guidance_label = QLabel("Guidance Scale:")
        embedded_guidance_label.setToolTip(guidance_tip)
        self.embedded_guidance_value_label = QLabel(str(self.global_settings.embedded_guidance))
        self.embedded_guidance_value_label.setToolTip(guidance_tip)
        self.embedded_guidance_slider = QSlider(Qt.Horizontal)
        self.embedded_guidance_slider.setToolTip(guidance_tip)
        self.embedded_guidance_slider.setMinimum(10)  # Internal integer range 10 to 300 gives a float range of 1.0 to 30.0 when divided by 10
        self.embedded_guidance_slider.setMaximum(300)
        self.embedded_guidance_slider.setValue(self.global_settings.embedded_guidance * 10)
        self.embedded_guidance_slider.setMinimumWidth(160)
        self.embedded_guidance_slider.valueChanged.connect(lambda value: self.global_settings.update("embedded_guidance", value / 10, self.embedded_guidance_value_label, value / 10))

        # --- Flow Shift Slider ---
        flow_shift_tip = "Flow shift: Affects distribution of denoising timesteps. Generally lower amounts of steps need more. Default is 7.0"
        flow_shift_label = QLabel("Flow Shift:")
        flow_shift_label.setToolTip(flow_shift_tip)
        self.flow_shift_value_label = QLabel(str(self.global_settings.flow_shift))
        self.flow_shift_value_label.setToolTip(flow_shift_tip)
        self.flow_shift_slider = QSlider(Qt.Horizontal)
        self.flow_shift_slider.setToolTip(flow_shift_tip)
        self.flow_shift_slider.setMinimum(10)
        self.flow_shift_slider.setMaximum(300)
        self.flow_shift_slider.setValue(self.global_settings.flow_shift * 10)
        self.flow_shift_slider.setMinimumWidth(160)
        self.flow_shift_slider.valueChanged.connect(lambda value: self.global_settings.update("flow_shift", value / 10, self.flow_shift_value_label, value / 10))

        # --- Inference Steps Slider ---
        inference_steps_tip = "Number of inference steps, default is 50."
        inference_steps_label = QLabel("Steps:")
        inference_steps_label.setToolTip(inference_steps_tip)
        self.inference_steps_value_label = QLabel(str(self.global_settings.infer_steps))
        self.inference_steps_value_label.setToolTip(inference_steps_tip)
        self.inference_steps_slider = QSlider(Qt.Horizontal)
        self.inference_steps_slider.setMinimum(1)
        self.inference_steps_slider.setMaximum(200)
        self.inference_steps_slider.setValue(self.global_settings.infer_steps)
        self.inference_steps_slider.setMinimumWidth(160)
        self.inference_steps_slider.setToolTip(inference_steps_tip)
        self.inference_steps_slider.valueChanged.connect(lambda value: self.global_settings.update("infer_steps", value, self.inference_steps_value_label, value))

        slider_layout.addWidget(embedded_guidance_label)
        slider_layout.addWidget(self.embedded_guidance_slider)
        slider_layout.addWidget(self.embedded_guidance_value_label, alignment=Qt.AlignRight)
        slider_layout.addWidget(flow_shift_label)
        slider_layout.addWidget(self.flow_shift_slider)
        slider_layout.addWidget(self.flow_shift_value_label, alignment=Qt.AlignRight)
        slider_layout.addWidget(inference_steps_label)
        slider_layout.addWidget(self.inference_steps_slider)
        slider_layout.addWidget(self.inference_steps_value_label, alignment=Qt.AlignRight)
        slider_group.setLayout(slider_layout)
        second_row_layout.addWidget(slider_group)

        # --- Block Swap Dial
        dial_layout = QVBoxLayout()
        block_swap_label = QLabel("Blocks to swap")
        dial_tip = "blocks_to_swap: Number of transformer blocks to offload to CPU. Higher saves VRAM and avoids OOM at cost of speed."
        block_swap_label.setToolTip(dial_tip)
        block_swap_dial = ValueDial(minimum=0, maximum=37)
        block_swap_dial.setValue(self.global_settings.blocks_to_swap)
        block_swap_dial.setNotchesVisible(True)
        block_swap_dial.valueChanged.connect(lambda value: self.global_settings.update("blocks_to_swap", value))
        block_swap_dial.setMinimumSize(120, 120)
        block_swap_dial.setToolTip(dial_tip)
        dial_layout.addWidget(block_swap_dial)
        dial_layout.addWidget(block_swap_label, alignment=Qt.AlignHCenter)
        dial_layout.addStretch()

        attention_layout = QHBoxLayout()
        attention_tip = "Attention: Selects the attention implementation to use for inference. The default, sage, is fastest and most memory efficient."
        attention_label = QLabel("Attention:")
        attention_label.setToolTip(attention_tip)
        self.attention_combo = QComboBox()
        self.attention_combo.addItems(["flash", "torch", "sdpa", "sage", "xformers"])
        self.attention_combo.setCurrentText(self.global_settings.attention)
        self.attention_combo.currentTextChanged.connect(lambda value: self.global_settings.update("attention", value))
        self.attention_combo.setToolTip(attention_tip)
        attention_layout.addWidget(attention_label)
        attention_layout.addWidget(self.attention_combo)
        dial_layout.addLayout(attention_layout)

        # Button that will open the settings dialog.
        open_settings_button = QPushButton("Open Settings")
        open_settings_button.clicked.connect(self.open_settings)
        dial_layout.addWidget(open_settings_button)
        second_row_layout.addLayout(dial_layout)

        second_row_right = QVBoxLayout()

        secondary_options_group = QGroupBox()
        secondary_options_group.setMinimumWidth(100)
        secondary_options_group_layout = QHBoxLayout()
        seed_tip = "Seed: Seed to use for initializing noise. If 'Reproducible' is enabled in LLM options then same seed, same prompt, same result"
        self.seed_edit = SeedWidget(self.global_settings)
        self.seed_edit.setToolTip(seed_tip)
        secondary_options_group_layout.addWidget(self.seed_edit)
        secondary_options_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        secondary_options_group.setLayout(secondary_options_group_layout)
        second_row_right.addWidget(secondary_options_group)

        # --- LLM Options Box ---
        llm_options_group = QGroupBox()
        llm_options_labeling_layout = QVBoxLayout()
        llm_options_labeling_layout.addWidget(QLabel("LLM Options:"), alignment=Qt.AlignCenter)
        llm_options_layout = QHBoxLayout()
        hssl_tip = "hidden_state_skip_layer: Number of layers to skip when processing the hidden state. Default 2."
        hssl_label = QLabel("Layer skip:")
        hssl_label.setToolTip(hssl_tip)
        self.hssl_input = QLineEdit()
        self.hssl_input.setToolTip(hssl_tip)
        self.hssl_input.setText(str(self.global_settings.hidden_state_skip_layer))
        self.hssl_input.setFixedWidth(30)
        self.hssl_input.setValidator(QIntValidator(0, 5, self))
        self.hssl_input.textChanged.connect(lambda text: self.global_settings.update("hidden_state_skip_layer", text))

        self.afn_checkbox = QCheckBox("Final Norm")
        self.afn_checkbox.setToolTip("apply_final_norm: Whether to apply the final normalization step hidden states.")
        self.afn_checkbox.setChecked(self.global_settings.apply_final_norm)
        self.afn_checkbox.toggled.connect(lambda checked: self.global_settings.update("apply_final_norm", checked))

        self.reproduce_checkbox = QCheckBox("Reproducible")
        self.reproduce_checkbox.setToolTip("reproduce: Whether to make the LLM output reproducible and thus same seed = same output")
        self.reproduce_checkbox.setChecked(self.global_settings.reproduce)
        self.reproduce_checkbox.toggled.connect(lambda checked: self.global_settings.update("reproduce", checked))

        llm_options_layout.addStretch()
        llm_options_layout.addWidget(hssl_label, alignment=Qt.AlignLeft)
        llm_options_layout.addWidget(self.hssl_input, alignment=Qt.AlignLeft)
        llm_options_layout.addStretch()
        llm_options_layout.addWidget(self.afn_checkbox, alignment=Qt.AlignLeft)
        llm_options_layout.addStretch()
        llm_options_layout.addWidget(self.reproduce_checkbox, alignment=Qt.AlignLeft)
        llm_options_layout.addStretch()
        llm_options_labeling_layout.addLayout(llm_options_layout)
        llm_options_group.setLayout(llm_options_labeling_layout)
        second_row_right.addWidget(llm_options_group)

        second_row_layout.addLayout(second_row_right)
        main_layout.addLayout(second_row_layout)

        # --- Checkboxes for Options ---
        checkbox_group = QGroupBox("Options")
        checkbox_layout = QHBoxLayout()
        self.fp8_checkbox = QCheckBox("fp8")
        self.fp8_checkbox.setToolTip("fp8: If enabled, model will be loaded in fp8 precision saving significant VRAM at the cost of some quality.")
        self.fp8_checkbox.setChecked(self.global_settings.fp8)
        self.fp8_checkbox.toggled.connect(lambda checked: self.global_settings.update("fp8", checked))

        self.fp8_fast_checkbox = QCheckBox("fp8 fast")
        self.fp8_fast_checkbox.setToolTip("fp8 fast: Enable acceleration for fp8 arithmetic, supported on RTX 4xxx+")
        self.fp8_fast_checkbox.setChecked(self.global_settings.fp8_fast)
        self.fp8_fast_checkbox.toggled.connect(lambda checked: self.global_settings.update("fp8_fast", checked))

        self.do_compile_checkbox = QCheckBox("torch.compile")
        self.do_compile_checkbox.setToolTip("torch.compile: If enabled, torch.compile will be used to optimize the model before inference, gaining speed and reducing VRAM usage in exchange for a small startup delay ")
        self.do_compile_checkbox.setChecked(self.global_settings.do_compile)
        self.do_compile_checkbox.toggled.connect(lambda checked: self.global_settings.update("do_compile", checked))

        checkbox_layout.addWidget(self.fp8_checkbox)
        checkbox_layout.addWidget(self.fp8_fast_checkbox)
        checkbox_layout.addWidget(self.do_compile_checkbox)
        checkbox_group.setLayout(checkbox_layout)
        main_layout.addWidget(checkbox_group)

        self.generate_button = QPushButton("Generate!")
        self.generate_button.clicked.connect(self.generate_video)
        main_layout.addWidget(self.generate_button)

        # --- Video Preview Widget ---
        preview_label = QLabel("Latent Preview:")
        main_layout.addWidget(preview_label)
        self.video_widget = QVideoWidget()
        # Ensure the video widget is visible and expands with the window
        self.video_widget.setMinimumSize(320, 240)
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.video_widget)
        self.media_player = QMediaPlayer()
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.setLoops(-1)
        self.media_player.errorOccurred.connect(self.handle_error)

        video_file = "latent_preview.mp4"
        if os.path.exists(video_file):
            video_url = QUrl.fromLocalFile(os.path.abspath(video_file))
            self.media_player.setSource(video_url)
            self.media_player.play()
        else:
            print("Video file not found:", video_file)

        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self.update_preview)
        self.preview_timer.start(1000)

    def open_settings(self):
        """Create and show the settings dialog."""
        dialog = SettingsDialog(self)
        dialog.exec()

    def update_preview(self):
        if self.global_settings.generating == 1:
            if os.path.isfile("./previewflag"):
                print("Updating preview...")
                os.remove("./previewflag")
                if os.path.isfile(self.global_settings.last_preview_file):
                    os.remove(self.global_settings.last_preview_file)
                    self.global_settings.last_preview_file = ""
                    print("Remove old file!")
                salt = int(random.random() * 1024)
                video_file = f"latent_preview_copy_{salt}.mp4"
                shutil.copy("./latent_preview.mp4", video_file)
                self.global_settings.last_preview_file = video_file
                if os.path.exists(video_file):
                    video_url = QUrl.fromLocalFile(os.path.abspath(video_file))
                    self.media_player.setSource(video_url)
                    self.media_player.play()
                else:
                    print("Latent preview not found:", video_file)
                print("Preview updated!")

    def handle_error(self, error):
        print("Media player error:", self.media_player.errorString())

    def generate_video(self):
        def subprocess_runner(musubi_command):
            self.global_settings.generating = 1
            try:
                subprocess.run(musubi_command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print("Error calling Musubi: ", e)
            self.global_settings.generating = 0

        print("Generate!")
        self.global_settings.save_to_file()
        if self.global_settings.generating != 1:
            seed = self.global_settings.seed if self.global_settings.seed != -1 else random.randint(-999999999, 999999999)
            attention = self.global_settings.attention if self.global_settings.attention != "sage" else "sageattn"
            musubi_command = (f"python hv_generate_video.py --dit {self.global_settings.transformer_path} --text_encoder1 {self.global_settings.text_encoder_1_path} "
                              f"--text_encoder2 {self.global_settings.text_encoder_2_path} --seed {seed} "
                              f"--vae {self.global_settings.vae_path} --save_path ./ --blocks_to_swap 14 "
                              f"--flow_shift {self.global_settings.flow_shift} --hidden_state_skip_layer {self.global_settings.hidden_state_skip_layer} "
                              f"--video_size {self.global_settings.resolution_y} {self.global_settings.resolution_x} --fps {self.global_settings.fps} --video_length "
                              f"{self.global_settings.video_length} --infer_steps {self.global_settings.infer_steps} --embedded_cfg_scale {self.global_settings.embedded_guidance} "
                              f" --attn_mode {attention} --output_type latent --prompt '{self.global_settings.prompt}' --preview_latent_every 2  ")
            if self.global_settings.fp8:
                musubi_command += "--fp8 "
            if self.global_settings.fp8_fast:
                musubi_command += "--fp8_fast "
            if self.global_settings.do_compile:
                musubi_command += "--compile "
            if self.global_settings.reproduce:
                musubi_command += "--reproduce "
            if self.global_settings.apply_final_norm:
                musubi_command += "--apply_final_norm True"
            print(musubi_command)
            generation_thread = threading.Thread(target=subprocess_runner, args=(musubi_command, ), daemon=True)
            generation_thread.start()
        else:
            print("Already generating!")


def cleanup():
    print("Cleaning up before exit...")
    global_settings = BlissfulSettings()
    global_settings.save_to_file()
    if os.path.isfile(global_settings.last_preview_file):
        os.remove(global_settings.last_preview_file)


if __name__ == "__main__":
    if os.path.isfile("./previewflag"):
        os.remove("./previewflag")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.aboutToQuit.connect(cleanup)
    sys.exit(app.exec())
