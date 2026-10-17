# DLSS-NR Source Attribution

The packed-weight layout, fixed network profile, geometry, noise, preprocessing
and numerical operations in this package are adapted from the MIT-licensed
[OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR) project, commit
`9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1`.

Fixed condition encoding in `controls.py` follows that revision's
[`nr_preprocess.comp`](https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1/demo/shaders/nr_preprocess.comp).
These controls do not add an external segmentation model or change the loss mask.

Copyright (c) 2026 maan. The original copyright and permission notice is retained
in [LICENSE.OpenDLSS-NR](LICENSE.OpenDLSS-NR). The PyTorch training integration and
surrogate gradients are not an upstream OpenDLSS-NR implementation.

No NVIDIA DLLs, model weights, headers, documentation or game assets are
distributed with this package. Users must supply their own source weights and
have the rights required to use them. The source-code licenses do not grant any
rights to those external artifacts.

This implementation is not affiliated with or endorsed by NVIDIA Corporation.
DLSS is a trademark of NVIDIA Corporation and is used to identify the model
format. DLL output equivalence and native deployment have not been validated.

## DLL Resource I/O

The PE/WEIGHTS_HT resource format handling in `dll.py` is adapted from
[MLX-DLSS](https://github.com/sdbds/MLX-DLSS), commit
`0ca2deab092fe6f3e331bf4f616271dbc64521d0`, specifically
`python/mlxdlss/tools/extract_dlssnr_weights.py`.
MLX-DLSS, Copyright 2026 MLX-DLSS contributors, is licensed under Apache-2.0;
the license is retained in [LICENSE.MLX-DLSS](LICENSE.MLX-DLSS).
The adapted implementation adds bounded parsing and replaces only same-sized
payload ranges instead of reserializing the resource. It does not depend on MLX
or MLX Swift. The MLX logical-v18 tensor layout is not imported into the trainer.

The user's `roundtrip_probe.py` informed the original-byte verification contract.
Native export keeps the existing canonical layout and adds explicit mixing,
numeric-weight scaling, quantization and re-extraction checks. These checks do
not certify a modified DLL's signature, native loading, or rendered output.
