# Copyright (c) 2025 Z-Image Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified from the original version.
# Original implementation: https://github.com/Tongyi-MAI/Z-Image
# Modifications: Copied and modified for Musubi Tuner project.

# region Inference Configuration Constants

"""Inference-specific configuration for Z-Image."""

DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_INFERENCE_STEPS = 8
DEFAULT_GUIDANCE_SCALE = 0.0
DEFAULT_CFG_TRUNCATION = 1.0
DEFAULT_MAX_SEQUENCE_LENGTH = 512

# endregion

# region Z-Image Model Configuration Constants

"""Model configuration constants for Z-Image."""

ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32

ROPE_THETA = 256.0
ROPE_AXES_DIMS = [32, 48, 48]
ROPE_AXES_LENS = [1536, 512, 512]

FREQUENCY_EMBEDDING_SIZE = 256
MAX_PERIOD = 10000

BASE_IMAGE_SEQ_LEN = 256
MAX_IMAGE_SEQ_LEN = 4096
BASE_SHIFT = 0.5
MAX_SHIFT = 1.15

DEFAULT_VAE_SCALE_FACTOR = 8
DEFAULT_VAE_IN_CHANNELS = 3
DEFAULT_VAE_OUT_CHANNELS = 3
# DEFAULT_VAE_LATENT_CHANNELS = 4
ZIMAGE_VAE_LATENT_CHANNELS = 16
DEFAULT_VAE_NORM_NUM_GROUPS = 32
# DEFAULT_VAE_SCALING_FACTOR = 0.18215
ZIMAGE_VAE_SCALING_FACTOR = 0.3611
ZIMAGE_VAE_SHIFT_FACTOR = 0.1159
ZIMAGE_VAE_SCALE_FACTOR = 8  # 2 ** (len(block_out_channels) - 1)
DEFAULT_TRANSFORMER_PATCH_SIZE = (2,)
DEFAULT_TRANSFORMER_F_PATCH_SIZE = (1,)
DEFAULT_TRANSFORMER_IN_CHANNELS = 16
DEFAULT_TRANSFORMER_DIM = 3840
DEFAULT_TRANSFORMER_N_LAYERS = 30
DEFAULT_TRANSFORMER_N_REFINER_LAYERS = 2
DEFAULT_TRANSFORMER_N_HEADS = 30
DEFAULT_TRANSFORMER_N_KV_HEADS = 30
DEFAULT_TRANSFORMER_NORM_EPS = 1e-5
DEFAULT_TRANSFORMER_QK_NORM = True
DEFAULT_TRANSFORMER_CAP_FEAT_DIM = 2560
DEFAULT_TRANSFORMER_T_SCALE = 1000.0

DEFAULT_SCHEDULER_NUM_TRAIN_TIMESTEPS = 1000
DEFAULT_SCHEDULER_SHIFT = 3.0
DEFAULT_SCHEDULER_USE_DYNAMIC_SHIFTING = False

DEFAULT_LOAD_DEVICE = "cuda"
DEFAULT_LOAD_DTYPE_STR = "bfloat16"

BYTES_PER_GB = 2**30

# endregion
