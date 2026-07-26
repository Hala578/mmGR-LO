"""Command-line interface for the mmGR-LO physics generation module."""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .physics import (
    estimate_reference_angles,
    transformation_vector,
    warp_doppler_time_map,
)


def _load_vector(path: str) -> np.ndarray:
    source = Path(path)
    if source.suffix.lower() == ".npy":
        return np.asarray(np.load(source), dtype=np.float64).reshape(-1)
    return np.asarray(np.loadtxt(source, delimiter=","), dtype=np.float64).reshape(-1)


def _save_vector(path: str, values: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".npy":
        np.save(destination, values)
    else:
        np.savetxt(destination, values, delimiter=",")


def _load_dtm(path: str):
    source = Path(path)
    if source.suffix.lower() == ".npy":
        return np.load(source), None
    image = Image.open(source)
    return np.asarray(image), image.mode


def _save_dtm(path: str, values: np.ndarray, image_mode: Optional[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".npy":
        np.save(destination, values)
        return

    if image_mode is None:
        raise ValueError("use a .npy output for numerical DTM input")
    if np.issubdtype(values.dtype, np.integer):
        converted = values
    else:
        converted = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    Image.fromarray(converted, mode=image_mode).save(destination)


def _estimate(args: argparse.Namespace) -> None:
    signs = _load_vector(args.angle_signs) if args.angle_signs else None
    angles = estimate_reference_angles(
        _load_vector(args.doppler_hz),
        _load_vector(args.ranges_m),
        args.frame_interval_s,
        args.wavelength_m,
        angle_signs=signs,
        invalid=args.invalid,
    )
    _save_vector(args.output, angles)


def _generate(args: argparse.Namespace) -> None:
    dtm, image_mode = _load_dtm(args.input)
    if args.reference_angles:
        reference = _load_vector(args.reference_angles)
    else:
        reference = np.asarray(args.reference_angle_deg, dtype=np.float64)
    vector = transformation_vector(reference, args.angular_deviation_deg)
    doppler_axis = np.linspace(args.doppler_min, args.doppler_max, dtm.shape[0])
    generated = warp_doppler_time_map(dtm, doppler_axis, vector)
    _save_dtm(args.output, generated, image_mode)
    if args.save_vector:
        _save_vector(args.save_vector, np.atleast_1d(vector))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Physics-based DTM generation from Section IV-C.1 of mmGR-LO"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate = subparsers.add_parser(
        "estimate", help="estimate reference motion angles using Eq. (8)"
    )
    estimate.add_argument("--doppler-hz", required=True, help=".npy or CSV vector")
    estimate.add_argument("--ranges-m", required=True, help=".npy or CSV vector")
    estimate.add_argument("--frame-interval-s", required=True, type=float)
    estimate.add_argument("--wavelength-m", type=float, default=0.004)
    estimate.add_argument("--angle-signs", help="optional .npy/CSV vector of -1/+1")
    estimate.add_argument("--invalid", choices=("raise", "nan"), default="raise")
    estimate.add_argument("--output", required=True, help="output .npy or CSV path")
    estimate.set_defaults(handler=_estimate)

    generate = subparsers.add_parser(
        "generate", help="apply Eq. (9) to a numerical or image DTM"
    )
    generate.add_argument("--input", required=True, help="input .npy or image DTM")
    generate.add_argument("--output", required=True, help="output .npy or image DTM")
    angles = generate.add_mutually_exclusive_group(required=True)
    angles.add_argument("--reference-angles", help="per-frame .npy/CSV angles")
    angles.add_argument("--reference-angle-deg", type=float, help="one angle for all frames")
    generate.add_argument("--angular-deviation-deg", required=True, type=float)
    generate.add_argument("--doppler-min", type=float, default=-1.0)
    generate.add_argument("--doppler-max", type=float, default=1.0)
    generate.add_argument("--save-vector", help="optional Eq. (9) vector output")
    generate.set_defaults(handler=_generate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
