# mmGR-LO

Official implementation of **mmGR-LO: Advancing Location and
Orientation-Independent Sensing for Gesture Recognition Using mmWave**.

This release contains both stages of gesture feature generation:

1. the physics-based transformation that generates Doppler-time maps (DTMs)
   for new locations and orientations; and
2. the controlled diffusion sampler that enhances the generated DTMs.

## Repository Structure

```text
mmGR-LO/
  physics_generation/
    __init__.py
    physics.py
    cli.py
  tests/
    test_physics_generation.py
  checkpoints/
    ema_0.9999_028000.pt
  ref_imgs/
    pushtarget40/
    ...
    mid_push_1.5m_40_01_Raw_0.bin.jpg
    ...
  scripts/
    mmgr_sample.py
    resizer.py
    guided_diffusion/
  README.md
  LICENSE
  requirements.txt
```

## Environment

Python 3.8 or later is recommended.

```bash
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build first if GPU inference is required. The
physics module itself only requires NumPy and Pillow and runs on CPU.

## Physics-Based Feature Generation

The implementation in `physics_generation/physics.py` corresponds to Section
IV-C.1 and Eqs. (6)-(9) of the paper. It does not depend on plot colors or
manually selected contours.

### Inputs

The module accepts either a numerical DTM (`.npy`) or a DTM image. A DTM must
have shape `[doppler_bins, frames]` or `[doppler_bins, frames, channels]`.

To estimate the reference motion angle with Eq. (8), provide:

- `ranges_m`: `N + 1` consecutive target-to-radar ranges in metres;
- `doppler_hz`: `N` Doppler-frequency estimates, one per frame interval;
- `frame_interval_s`: the interval between consecutive range samples; and
- `wavelength_m`: the radar wavelength (`0.004` m by default).

The cosine rule determines the angle magnitude but not which side of the
radial baseline the motion occupies. For a signed nonlinear trajectory, also
provide an `N`-element vector containing `-1` or `+1`, following the paper's
counterclockwise-positive convention.

### Step 1: Estimate Reference Angles

Input vectors may be NumPy arrays (`.npy`) or comma-separated text files.

```bash
python -m physics_generation.cli estimate \
  --ranges-m data/reference_ranges.npy \
  --doppler-hz data/reference_doppler_hz.npy \
  --frame-interval-s 0.08 \
  --wavelength-m 0.004 \
  --angle-signs data/reference_angle_signs.npy \
  --output outputs/reference_angles.npy
```

Omit `--angle-signs` for a trajectory whose principal angles in `[0, 180]`
degrees are sufficient. By default, physically inconsistent range/Doppler
samples raise an error instead of silently creating invalid training data.

### Step 2: Generate a Target-Configuration DTM

`angular-deviation-deg` is the combined location and execution-orientation
deviation, `theta_n`, in Eq. (9).

```bash
python -m physics_generation.cli generate \
  --input data/reference_dtm.npy \
  --reference-angles outputs/reference_angles.npy \
  --angular-deviation-deg 30 \
  --doppler-min -2.77 \
  --doppler-max 2.77 \
  --save-vector outputs/transformation_vector.npy \
  --output outputs/physics_dtm_30deg.npy
```

For a DTM image and one constant reference angle:

```bash
python -m physics_generation.cli generate \
  --input data/reference_dtm.png \
  --reference-angle-deg 0 \
  --angular-deviation-deg 30 \
  --output outputs/physics_dtm_30deg.png
```

The generated image or array is the physics-based feature used as the start
image for the controlled diffusion enhancement stage. A negative component
of the transformation vector automatically reverses the Doppler direction,
as described for nonlinear gestures in the paper.

### Python API

```python
import numpy as np

from physics_generation import transformation_vector, warp_doppler_time_map

dtm = np.load("data/reference_dtm.npy")
reference_angles = np.load("outputs/reference_angles.npy")
doppler_axis = np.linspace(-2.77, 2.77, dtm.shape[0])

vector = transformation_vector(reference_angles, angular_deviation_deg=30)
generated_dtm = warp_doppler_time_map(dtm, doppler_axis, vector)
np.save("outputs/physics_dtm_30deg.npy", generated_dtm)
```

### Verify the Physics Module

The tests cover Eq. (8), Eq. (9), identity generation, and Doppler sign
reversal:

```bash
python -m unittest discover -s tests -v
```

## Controlled Diffusion Enhancement

The `ref_imgs/` directory contains reference-image folders for
`--base_samples` and packaged `mid_*.jpg` start images. Available gesture
categories are `push`, `pull`, `slide`, `sweep`, `kock`, and `zigzag`, with
target levels from `40` to `140` where provided.

Run the sampler from the repository root:

```bash
python scripts/mmgr_sample.py \
  --model_path checkpoints/ema_0.9999_028000.pt \
  --base_samples ref_imgs/pushtarget90 \
  --start_image outputs/physics_dtm_30deg.png \
  --save_dir outputs/push_demo \
  --attention_resolutions 16 \
  --class_cond False \
  --diffusion_steps 500 \
  --dropout 0.0 \
  --image_size 128 \
  --learn_sigma True \
  --noise_schedule linear \
  --num_channels 128 \
  --num_head_channels 64 \
  --num_res_blocks 1 \
  --resblock_updown True \
  --use_fp16 False \
  --use_scale_shift_norm True \
  --timestep_respacing 100 \
  --down_N 2 \
  --range_t 5 \
  --batch_size 1 \
  --num_samples 1
```

`--base_samples` must be a directory. `--start_image` must be an image file
and may be either a physics-generated DTM or a packaged `mid_*.jpg` image.
`--down_N` must be a power of two. Add `--save_intermediate` to save reverse
diffusion steps under `<save_dir>/steps/`.

Final generated images are written to `--save_dir` as `sample_0000.png`,
`sample_0001.png`, and so on. CPU inference is supported but considerably
slower than GPU inference.

## Reproducibility Notes

- The physics generator expects the DTM time axis to be the second dimension.
- Use the Doppler limits associated with the radar configuration that produced
  the reference DTM.
- Eq. (9) is singular when the reference motion is perpendicular to the radar
  radial direction (`cos(theta_i)` is approximately zero); the implementation
  reports this condition explicitly.
- Keep the action category and target level of `--base_samples` consistent
  with the generated start image.
- The repository checkpoint is tracked with Git LFS.

## License

This project is released under the MIT License. See `LICENSE` for details.
