# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from easydict import EasyDict

from musubi_tuner.wan.configs.shared_config import wan_shared_cfg

# ------------------------ Wan T2V 1.3B ------------------------#

t2v_1_3B = EasyDict(__name__="Config: Wan T2V 1.3B")
t2v_1_3B.update(wan_shared_cfg)
t2v_1_3B.i2v = False
t2v_1_3B.is_fun_control = False
t2v_1_3B.flf2v = False
t2v_1_3B.v2_2 = False

# t5
t2v_1_3B.t5_checkpoint = "models_t5_umt5-xxl-enc-bf16.pth"
t2v_1_3B.t5_tokenizer = "google/umt5-xxl"

# vae
t2v_1_3B.vae_checkpoint = "Wan2.1_VAE.pth"
t2v_1_3B.vae_stride = (4, 8, 8)

# transformer
t2v_1_3B.patch_size = (1, 2, 2)
t2v_1_3B.dim = 1536
t2v_1_3B.ffn_dim = 8960
t2v_1_3B.freq_dim = 256
t2v_1_3B.in_dim = 16
t2v_1_3B.num_heads = 12
t2v_1_3B.num_layers = 30
t2v_1_3B.window_size = (-1, -1)
t2v_1_3B.qk_norm = True
t2v_1_3B.cross_attn_norm = True
t2v_1_3B.eps = 1e-6

# inference
t2v_1_3B.sample_shift = 5.0
t2v_1_3B.sample_steps = 50
t2v_1_3B.boundary = None
t2v_1_3B.sample_guide_scale = (5.0,)
