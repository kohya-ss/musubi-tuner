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
