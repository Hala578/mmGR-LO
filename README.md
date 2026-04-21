# mmGR-LO

## Overview
This repository is a lightweight public release of mmGR-LO. It includes the files required for inference reproduction: the trained inference checkpoint, the complete packaged reference-image set, a collection of mid-step start images, the inference code, and step-by-step instructions.

## Directory Structure
```text
mmGR-LO/
  README.md
  README_zh.md
  LICENSE
  requirements.txt
  checkpoints/
    ema_0.9999_028000.pt
  ref_imgs/
    pushtarget40/
      push_1.5m_40_07_Raw_0.bin.jpg
    ...
    zigzagtarget140/
      zigzag_1.5m_140_07_Raw_0.bin.jpg
    mid_push_1.5m_40_01_Raw_0.bin.jpg
    ...
    mid_zigzag_1.5m_140_01_Raw_0.bin.jpg
  scripts/
    mmgr_sample.py
    resizer.py
    __init__.py
    guided_diffusion/
```

## Environment Setup
We recommend Python 3.8 or later.

1. Enter the release folder:
```bash
cd mmGR-LO
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. If you need a CUDA-enabled PyTorch build, install the matching PyTorch version first, then run the command above.

Notes:
- This release includes the trained inference checkpoint `checkpoints/ema_0.9999_028000.pt`.
- This release supports direct single-process inference by default.
- You do not need MPI or `mpiexec`.
- CPU inference is possible, but it will be much slower than GPU inference.

## Reference Images and Start Images
The `ref_imgs/` folder now contains two kinds of inputs:

- Reference-image folders for `--base_samples`. Each folder contains the reference image(s) for one action and target combination, such as `ref_imgs/pushtarget100` or `ref_imgs/zigzagtarget60`.
- Mid-step start images for `--start_image`. These are the `mid_*.jpg` files stored directly under `ref_imgs/`, such as `ref_imgs/mid_push_1.5m_100_01_Raw_0.bin.jpg`.

Packaged action categories:
- `push`
- `pull`
- `slide`
- `sweep`
- `kock`
- `zigzag`

Reference-image target levels currently packaged:
- `40`
- `50`
- `60`
- `70`
- `80`
- `90`
- `100`
- `110`
- `120`
- `130`
- `140`

Mid-step start-image target levels currently packaged:
- `40`
- `50`
- `60`
- `70`
- `80`
- `100`
- `110`
- `120`
- `130`
- `140`

Usage guidance:
- Use `--base_samples` to select the reference folder for the desired action and target level.
- Use `--start_image` only when you want to start sampling from a packaged mid-step image or your own custom image.
- Matching the action category and target level between `--base_samples` and `--start_image` is recommended.
- The packaged reference folders are intended for single-image inference, so `--batch_size 1` is recommended.

## Quick Start
Run the following command from the `mmGR-LO/` root directory. The example below uses the packaged `pushtarget90` reference folder and saves results to `outputs/push_demo/`.

```bash
python scripts/mmgr_sample.py --model_path checkpoints/ema_0.9999_028000.pt --base_samples ref_imgs/pushtarget90 --save_dir outputs/push_demo --attention_resolutions 16 --class_cond False --diffusion_steps 500 --dropout 0.0 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 1 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --timestep_respacing 100 --down_N 2 --range_t 5 --batch_size 1 --num_samples 1
```

If you want to start from a packaged mid-step image, use `--start_image` with a matching `mid_*.jpg` file. Example:

```bash
python scripts/mmgr_sample.py --model_path checkpoints/ema_0.9999_028000.pt --base_samples ref_imgs/pushtarget100 --start_image ref_imgs/mid_push_1.5m_100_01_Raw_0.bin.jpg --save_dir outputs/push100_midstart_demo --attention_resolutions 16 --class_cond False --diffusion_steps 500 --dropout 0.0 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 1 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --timestep_respacing 100 --down_N 2 --range_t 5 --batch_size 1 --num_samples 1
```

If you also want to save intermediate diffusion steps, append:

```bash
--save_intermediate
```

## Parameter Summary
Required arguments:
- `--model_path`: path to the trained inference checkpoint
- `--base_samples`: path to the reference image directory under `ref_imgs/`
- `--save_dir`: output directory

Optional arguments:
- `--start_image`: packaged mid-step start image under `ref_imgs/` or your own custom initial image
- `--save_intermediate`: save diffusion step images to `<save_dir>/steps/`

Default example model settings:
- `--image_size 128`
- `--num_channels 128`
- `--num_head_channels 64`
- `--num_res_blocks 1`
- `--attention_resolutions 16`
- `--resblock_updown True`
- `--learn_sigma True`
- `--noise_schedule linear`
- `--use_scale_shift_norm True`

Default example sampling settings:
- `--diffusion_steps 500`
- `--timestep_respacing 100`
- `--down_N 2`
- `--range_t 5`
- `--batch_size 1`
- `--num_samples 1`

Important:
- `--down_N` must be a power of 2.
- Use a folder path for `--base_samples` and a file path for `--start_image`.

## Switching Between Reference Categories
To switch categories, replace the `--base_samples` path with another packaged folder under `ref_imgs/`.

Examples:
- `ref_imgs/pushtarget40`
- `ref_imgs/pulltarget120`
- `ref_imgs/slidetarget90`
- `ref_imgs/sweeptarget70`
- `ref_imgs/kocktarget140`
- `ref_imgs/zigzagtarget50`

Example for `pushtarget40`:

```bash
python scripts/mmgr_sample.py --model_path checkpoints/ema_0.9999_028000.pt --base_samples ref_imgs/pushtarget40 --save_dir outputs/push40_demo --attention_resolutions 16 --class_cond False --diffusion_steps 500 --dropout 0.0 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 1 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --timestep_respacing 100 --down_N 2 --range_t 5 --batch_size 1 --num_samples 1
```

## Output Files
Default output behavior:
- Final generated images are saved under `--save_dir`
- File names follow the format `sample_0000.png`, `sample_0001.png`, and so on

If `--save_intermediate` is enabled:
- Intermediate diffusion step images are saved under `<save_dir>/steps/`

## License
This release is distributed under the MIT License. See `LICENSE` for details.
