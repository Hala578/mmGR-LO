import argparse
import math
import os


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"yes", "true", "t", "y", "1"}:
        return True
    if value in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


def add_defaults_to_argparser(parser, default_dict):
    for key, value in default_dict.items():
        value_type = type(value)
        if value is None:
            value_type = str
        elif isinstance(value, bool):
            value_type = str2bool
        parser.add_argument(f"--{key}", default=value, type=value_type)


def load_reference(data_dir, batch_size, image_size, class_cond=False):
    from guided_diffusion.image_datasets import load_data

    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
        deterministic=True,
        random_flip=False,
    )
    for large_batch, model_kwargs in data:
        model_kwargs["ref_img"] = large_batch
        yield model_kwargs


def load_start_image(path, image_size, batch_size, device):
    import torchvision as thv
    from PIL import Image

    image = Image.open(path).convert("RGB")
    preprocess = thv.transforms.Compose(
        [
            thv.transforms.Resize((image_size, image_size)),
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image_tensor = preprocess(image).unsqueeze(0)
    if batch_size > 1:
        image_tensor = image_tensor.repeat(batch_size, 1, 1, 1)
    return image_tensor.to(device)


def save_final_samples(sample, output_dir, start_index, count):
    from torchvision import utils

    for offset in range(count):
        out_path = os.path.join(output_dir, f"sample_{start_index + offset:04d}.png")
        utils.save_image(
            sample[offset].unsqueeze(0),
            out_path,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )


def mark_args_as_required(parser, arg_names):
    required_names = set(arg_names)
    for action in parser._actions:
        if action.dest in required_names:
            action.required = True


def main():
    args = create_argparser().parse_args()

    if args.down_N < 1 or not math.log(args.down_N, 2).is_integer():
        raise ValueError("--down_N must be a positive power of 2.")

    import torch.distributed as dist

    from guided_diffusion import dist_util, logger
    from guided_diffusion.script_util import (
        create_model_and_diffusion,
        model_and_diffusion_defaults,
    )
    from resizer import Resizer

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    logger.log("creating models...")
    model_keys = model_and_diffusion_defaults().keys()
    model, diffusion = create_model_and_diffusion(
        **{key: getattr(args, key) for key in model_keys}
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("creating resizers...")
    shape = (args.batch_size, 3, args.image_size, args.image_size)
    shape_d = (
        args.batch_size,
        3,
        int(args.image_size / args.down_N),
        int(args.image_size / args.down_N),
    )
    down = Resizer(shape, 1 / args.down_N).to(next(model.parameters()).device)
    up = Resizer(shape_d, args.down_N).to(next(model.parameters()).device)
    resizers = (down, up)

    logger.log("loading reference images...")
    data = load_reference(
        args.base_samples,
        args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    start_image = None
    if args.start_image:
        start_image = load_start_image(
            args.start_image,
            args.image_size,
            args.batch_size,
            dist_util.dev(),
        )

    logger.log("creating samples...")
    saved = 0
    intermediate_dir = (
        os.path.join(args.save_dir, "steps") if args.save_intermediate else None
    )
    while saved < args.num_samples:
        model_kwargs = next(data)
        model_kwargs = {key: value.to(dist_util.dev()) for key, value in model_kwargs.items()}
        sample = diffusion.p_sample_loop(
            model,
            shape,
            noise=start_image,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            resizers=resizers,
            range_t=args.range_t,
            save_path=intermediate_dir,
        )

        current_count = min(args.batch_size, args.num_samples - saved)
        save_final_samples(sample, logger.get_dir(), saved, current_count)
        saved += current_count
        logger.log(f"created {saved} samples")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        learn_sigma=True,
        image_size=128,
        num_channels=128,
        num_res_blocks=1,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=64,
        attention_resolutions="16",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
        diffusion_steps=500,
        noise_schedule="linear",
        timestep_respacing="100",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        down_N=2,
        range_t=5,
        use_ddim=False,
        base_samples=None,
        model_path=None,
        save_dir=None,
        start_image=None,
    )
    parser = argparse.ArgumentParser(
        description="Run mmGR-LO mmWave sampling with the released checkpoint."
    )
    add_defaults_to_argparser(parser, defaults)
    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        help="Save intermediate diffusion steps to <save_dir>/steps.",
    )
    mark_args_as_required(parser, {"model_path", "base_samples", "save_dir"})
    return parser


if __name__ == "__main__":
    main()
