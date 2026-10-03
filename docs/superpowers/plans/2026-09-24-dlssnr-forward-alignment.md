# DLSS-NR Forward Alignment Implementation Plan

> Execute locally in this task; the user prohibits subagents. Use systematic
> debugging, test-driven development and verification-before-completion.

**Goal:** Audit actual DLL defaults, correct the trainable forward's confirmed
semantic differences, and measure remaining disagreement without granting an
unsupported native-parity certificate.

**Architecture:** Keep canonical FP32 weights and the existing model/trainer
boundaries. Numerical primitives belong in `dlssnr/numerics.py`; model modules
call them without changing checkpoint keys. Preserve old experiment outputs.
Parameter tracing belongs only in the isolated external comparison harness.

**Spec:** `docs/dlssnr_310_8_0_musubi_tuner_spec_v0.2.md` and the user's request to
first exclude default-parameter differences, then restore the DLL's behavior.

## Constraints and Review Focus

- No subagents, commits, driver edits or changes to the supplied DLL.
- Do not relax the existing training/forward-validation gate.
- Check activation values outside its clamped polynomial interval.
- All FFN families must use the same intended activation, including ViT.
- Keep gradients finite and keep frozen LoRA base weights unchanged.
- A parameter report must distinguish omitted settings from explicit values.
- Style/intensity composition must not hide a raw-network mismatch.

## Tasks

- [x] Trace actual NGX Get calls and compare omitted controls with explicit ones
  on all four screenshots. Result: implicit controls match the explicit
  auto-mask-disabled tuple bit-for-bit; float getters are used for strengths,
  an unsigned getter for Style. Prior comparisons used explicit matching values.
- [x] Isolate the activation: replacing ordinary SiLU with the source's
  magnitude-preserving cubic lowers face MAE from 0.02936 to 0.00899.
- [x] Add scalar-value, derivative and all-FFN regression tests. Expected
  polynomial: `b=clamp(x,-4,4); x*(b*(-0.055908203125*abs(b)+0.447265625)+0.89453125)`.
  At x=1 it returns 1.285888671875, not ordinary SiLU's 0.7310586.
- [x] Implement the FP32 cubic in `numerics.py`, route DenseFFN, ExpertFFN and
  Split512FFN through it, and change the published numerical identity.
- [x] Repeat attention/publication diagnostics with the corrected activation.
  Retain only corrections demonstrated by reference evidence and regression
  tests, not image-tuned constants. Record unresolved arithmetic differences.
- [x] Re-run the screenshot/parameter matrix, LoRA/full CUDA smoke tests and the
  complete project test suite. Update documentation with measured agreement and
  explicit limitations; do not set a native validation flag from visual review.

Outcome: 172 project tests pass, including the real-weight CUDA tests. The
arithmetic version now has explicit scalar half/E4 publications and STE, keeping
FP32 master weights/matmul accumulation. 9/19 display cases pass the unchanged
suggested gate (8/18 excluding the zero-intensity bypass); strict native/head,
temporal and export acceptance remains incomplete. See the alignment report.
