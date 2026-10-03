# DLSS-NR Screenshot Forward Experiment, 2026-09-24

## Outcome

The current `train_surrogate` is **not equivalent** to the supplied DLSS-NR
310.8.0 DLL on the tested screenshots. This is a failed parity experiment, not
a successful `dlssnr_forward_validation_v1` report. Training's validation gate
remained closed at the time. No model/trainer/numerics code was changed in this experiment.
The current [training contract](dlssnr.md#训练输入与验证证据) no longer requires this
report to start training; that does not turn this experiment into passing evidence.

The DLL's default-style output visibly changes skin, surface detail and lighting.
The current FP32 network output stays much closer to the input. Adding the
Natural/Cinematic output grade does not close this gap.

All local inputs, outputs, scripts, arrays and measurements are under
`references/dlssnr_forward_20260924/`. This directory is gitignored. It contains
copyrighted publisher images and the user-supplied runtime; do not distribute the
directory as project source. The report below is the durable project record.

## Reference Identity

- GPU: NVIDIA GeForce RTX 4090; driver 610.88; PyTorch 2.13.0+cu130.
- DLL: `D:/UGit/OpenDLSS-NR/DLSS5VK_MODEL/nvngx_dlssnr.dll`, 310.8.0.0.
- DLL SHA-256: `984bee0f775c277d5829b8fd6775d53a7b0f75396c852b3aaf06a18375f81014`.
- Authenticode: `HashMismatch`. The user states it is a GPU-model compatibility
  patch only, with unchanged visual behavior. An unmodified signed DLL was not
  independently compared, so the precise claim is agreement with this local DLL.
- Bridge: [ComfyUI-DLSS5-NR](https://github.com/lisitskyaa/ComfyUI-DLSS5-NR),
  commit `41dcdfa593cb61b6a98c65bb8ed27606260bb598`, compiled locally with VS 2022.
- Local bridge changes: detect D3D12 device removal after completion; initialize
  the output texture with the input before evaluation. No runtime/driver patching.
- Native workers run in isolated processes. `ReleaseFeature` hangs with this
  runtime/bridge combination; after completed GPU readback and closed output
  files, workers terminate without invoking that release path. This harness is
  not a production integration.
- All **153 packed source records** occur byte-for-byte inside the local DLL.
  This establishes shared packed payloads, not correctness of tensor unpacking,
  channel interpretation, or forward arithmetic.
- Canonical SHA-256: `c6c0a4bcbc2e07c9c7c9f808212683e4682a42b48cbac828ec816ad602c659c0`.
- Canonical conversion round-trip: byte-identical; 145,755,115 logical parameters.
- `experiment_metadata.json` records further hashes and implementation identity.

## Inputs and Controls

Downloaded 12 publisher screenshots using Steam's store app-details API; selected
four covering a vehicle/interior, a face, a landscape and a village:

| Local ID | Publisher Store Page | Screenshot ID |
| --- | --- | --- |
| `cyberpunk_1` | [Cyberpunk 2077](https://store.steampowered.com/app/1091500/) | 1 |
| `witcher3_3` | [The Witcher 3](https://store.steampowered.com/app/292030/) | 3 |
| `witcher3_1` | [The Witcher 3](https://store.steampowered.com/app/292030/) | 1 |
| `kcd2_1` | [Kingdom Come: Deliverance II](https://store.steampowered.com/app/1771300/) | 1 |

`sources.json` retains direct image URLs, original dimensions and file hashes.
Originals are 1920x1080; both backends receive the same full-frame 960x540 Lanczos
resize, quantized towards zero to the bridge's FP16 upload grid. Our padded field
is 960x576. Inputs are SDR sRGB proxy code values, not linear HDR. RGB channel
order was fixed using an asymmetric color-ramp probe, not chosen separately for
each output. These are publisher screenshots, not raw engine-buffer captures.

19 image/control combinations per backend were tested. Each native call was also
repeated with reset enabled; all 19 repeats were bit-identical. Baseline settings:

```text
style=0, intensity=1, tone=1, structure=1, skin=-1, auto_mask=true
preset=3, reset=1, temporal=false, UI correction=0, scaling=1
```

Styles 0/1/2 were tested on all four images. The face image additionally tested
tone=0, structure=0, both=0, auto_mask=false, skin=0, intensity=0.5 and intensity=0.
No motion, depth or explicit UI/control mask was supplied. Publisher logos remain
part of the input; this experiment does not assess UI preservation.

Our frame seed is explicitly 0. The bridge does not expose the DLL's internal
seed, and input features were not captured. Identical reset repeats establish
determinism of this experiment, not matching random lanes between implementations.

## Parameter Findings

The [official report](https://research.nvidia.com/labs/adlr/files/DLSS5_Report.pdf),
pages 8-11, describes appearance selection, learned tone/structure controls, and
manual/semantic masks. It does **not** specify this private DLL's numerical style
IDs, ABI or all preprocessing/postprocessing details. Those details below follow
the [OpenDLSS-NR implementation](https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1/docs/frame.md)
and local measurements, not a published NVIDIA SDK contract.

The experiment encodes network lanes 10-14 as:

```text
auto_mask=true:  [style/128, tone, 1, skin if skin>=0 else structure, structure]
auto_mask=false: [style/128, tone, structure, -1, -1]
```

- Style 0/1/2 maps to Default/Natural/Cinematic in the community bridge. Style
  conditions the network; Natural/Cinematic also apply an output grade. The
  experiment preserves both raw network images and a separately graded display
  approximation. This grade is not integrated into production inference.
- Intensity blends the transformed output with the input. The native destination
  must already contain that input. An empty output texture produced black at 0
  and an incorrectly dark result at 0.5. Source-prefilling fixes the harness:
  intensity=0 matches the uploaded source exactly; intensity=0.5 differs from
  `(source + intensity1_output)/2` by MAE `0.00002583`, max `0.00036621`.
  The zero-intensity identity is a caller-prefilled bypass, not proof of a correct
  neural forward.
- Tone=0 and structure=0 still affect different aspects of the native image.
  Both=0 is close to identity: source MAE `0.00088535`, not bit-exact identity.
- Auto mask and skin controls change the native output. They were not silently
  treated as ordinary all-zero or all-one conditioning maps.

## Quantitative Result

Metrics below compare current FP32 output with the local DLL, using full-image
RGB values in [0,1], default style, full intensity. PSNR measures agreement with
the DLL, not perceptual quality. Images have no ground-truth photographic target.

| Input | MAE vs DLL | PSNR vs DLL | FP32 MAE vs input | DLL MAE vs input |
| --- | ---: | ---: | ---: | ---: |
| Cyberpunk vehicle | 0.036306 | 25.76 dB | 0.001913 | 0.035797 |
| Witcher face | 0.029360 | 26.05 dB | 0.001467 | 0.029237 |
| Witcher landscape | 0.042008 | 25.76 dB | 0.002390 | 0.040761 |
| KCD II village | 0.034926 | 26.51 dB | 0.002639 | 0.034480 |

The default-style discrepancy exists without any style grading. It is much
larger than output half rounding. On the face, changing our seed from 0 to 123
did not materially alter the DLL mismatch. This is not evidence that arbitrary
seed changes can be ignored in a future strict parity test.

Diagnostic-only attention ablations did not resolve it:

| Face experiment | MAE vs DLL |
| --- | ---: |
| Existing FP32 softmax | 0.029360 |
| Global-score clamp only | 0.029495 |
| Window-score clamp only | 0.029415 |
| Both clamps | 0.029441 |
| Native-style exponential approximation only | 0.029448 |

These substitutions were not applied to production. The root cause remains
unlocalized; there is no evidence yet to blame only FP8 rounding, one softmax
formula, the hidden seed, or a particular layer. Exact intermediate boundaries
are required to distinguish graph/layout errors from accumulated arithmetic drift.

## Artifacts and Reproduction

Important files inside the local experiment directory:

- `comparison_default.jpg`, `comparison_natural.jpg`, `comparison_cinematic.jpg`:
  input / local DLL / current FP32 display comparison.
- `face_detail.png`: identical 1:1 face crops.
- `native_controls.jpg`: native parameter sweep.
- `native/*.npy`: FP32 readbacks of native FP16 output textures.
- `surrogate/*.npz`: source, raw head, preclamp, rendered proxy and display arrays.
- `comparison_metrics.json`, `harness_verification.json`, `embedded_weights.json`.
- `experiment.py`, `attention_ablation.py`, `verify_experiment.py`.

After the local artifacts have been prepared, from the repository root:

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'src')
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
python references/dlssnr_forward_20260924/experiment.py surrogate
python references/dlssnr_forward_20260924/experiment.py native
python references/dlssnr_forward_20260924/experiment.py compare
python references/dlssnr_forward_20260924/verify_experiment.py
```

The harness checks pass; native parity does not. No passing forward-validation
report was generated, no training was started, and no project dependencies were
installed. The next step is to capture matching native features and layer outputs
and locate the earliest divergence before investing in full training. Still-image
results do not validate temporal reprojection, history publication or HDR behavior.
