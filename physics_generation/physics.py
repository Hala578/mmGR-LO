"""Implementation of the physical model in Section IV-C.1 of mmGR-LO.

The functions operate on numerical Doppler-time maps (DTMs), rather than
plot-specific colors or contours. This keeps the paper's transformation
explicit and allows the generated feature to be passed to the diffusion
enhancement stage or saved as an image.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


ArrayLike = Union[float, np.ndarray]


@dataclass(frozen=True)
class MotionGeometry:
    """Per-frame motion quantities recovered from range and Doppler tracks."""

    speed_m_s: np.ndarray
    radial_cosine: np.ndarray


def _as_1d_float(name: str, values: ArrayLike) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("{} must be a one-dimensional array".format(name))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} contains NaN or infinite values".format(name))
    return array


def estimate_motion_geometry(
    doppler_hz: ArrayLike,
    ranges_m: ArrayLike,
    frame_interval_s: float,
    wavelength_m: float,
    invalid: str = "raise",
) -> MotionGeometry:
    """Estimate speed and cos(theta_i) using Eqs. (5), (7), and (8).

    ``ranges_m`` contains the range at both ends of every frame interval, so
    it must have exactly one more element than ``doppler_hz``. With ``d_i``
    and ``d_(i+1)`` denoting consecutive ranges, the recovered speed is::

        v_i = sqrt(d_(i+1)^2 - d_i^2 + d_i f_di lambda dt) / dt

    and ``cos(theta_i) = f_di lambda / (2 v_i)``.

    Set ``invalid="nan"`` to mark intervals made physically inconsistent by
    measurement noise. The default raises an error so invalid inputs are not
    silently used to generate training data.
    """

    frequencies = _as_1d_float("doppler_hz", doppler_hz)
    ranges = _as_1d_float("ranges_m", ranges_m)
    if ranges.size != frequencies.size + 1:
        raise ValueError(
            "ranges_m must contain len(doppler_hz) + 1 samples; got {} and {}"
            .format(ranges.size, frequencies.size)
        )
    if frame_interval_s <= 0 or wavelength_m <= 0:
        raise ValueError("frame_interval_s and wavelength_m must be positive")
    if invalid not in {"raise", "nan"}:
        raise ValueError("invalid must be either 'raise' or 'nan'")

    d_i = ranges[:-1]
    d_next = ranges[1:]
    radicand = (
        np.square(d_next)
        - np.square(d_i)
        + d_i * frequencies * wavelength_m * frame_interval_s
    )
    tolerance = np.finfo(np.float64).eps * np.maximum(1.0, np.square(d_i)) * 32
    bad = radicand < -tolerance
    radicand = np.where(np.abs(radicand) <= tolerance, 0.0, radicand)

    if np.any(bad) and invalid == "raise":
        indices = np.flatnonzero(bad).tolist()
        raise ValueError(
            "range/Doppler samples are physically inconsistent at intervals {}"
            .format(indices)
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.sqrt(np.where(bad, np.nan, radicand)) / frame_interval_s
        cosine = frequencies * wavelength_m / (2.0 * speed)

    stationary = np.isclose(speed, 0.0)
    cosine = np.where(stationary & np.isclose(frequencies, 0.0), 1.0, cosine)
    out_of_range = np.abs(cosine) > 1.0 + 1e-7
    if np.any(out_of_range) and invalid == "raise":
        indices = np.flatnonzero(out_of_range).tolist()
        raise ValueError(
            "estimated cos(theta) falls outside [-1, 1] at intervals {}"
            .format(indices)
        )
    cosine = np.clip(cosine, -1.0, 1.0)
    cosine = np.where(bad | out_of_range, np.nan, cosine)
    return MotionGeometry(speed_m_s=speed, radial_cosine=cosine)


def estimate_reference_angles(
    doppler_hz: ArrayLike,
    ranges_m: ArrayLike,
    frame_interval_s: float,
    wavelength_m: float,
    angle_signs: Optional[ArrayLike] = None,
    invalid: str = "raise",
) -> np.ndarray:
    """Return reference angles in degrees for Eq. (9).

    The cosine rule determines the angle magnitude but not which side of the
    radial baseline the motion occupies. Pass per-frame ``angle_signs`` of
    ``-1`` or ``+1`` when that sign is known from the gesture trajectory.
    Without it, principal angles in ``[0, 180]`` degrees are returned.
    """

    geometry = estimate_motion_geometry(
        doppler_hz,
        ranges_m,
        frame_interval_s,
        wavelength_m,
        invalid=invalid,
    )
    angles = np.rad2deg(np.arccos(geometry.radial_cosine))
    if angle_signs is None:
        return angles

    signs = _as_1d_float("angle_signs", angle_signs)
    if signs.shape != angles.shape or not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("angle_signs must contain one -1 or +1 per interval")
    return angles * signs


def transformation_vector(
    reference_angles_deg: ArrayLike,
    angular_deviation_deg: ArrayLike,
    singular_tolerance: float = 1e-8,
) -> np.ndarray:
    """Construct ``T_i = cos(theta_i + theta_n) / cos(theta_i)`` (Eq. 9)."""

    reference = np.asarray(reference_angles_deg, dtype=np.float64)
    deviation = np.asarray(angular_deviation_deg, dtype=np.float64)
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(deviation)):
        raise ValueError("angles must be finite")

    denominator = np.cos(np.deg2rad(reference))
    if np.any(np.abs(denominator) < singular_tolerance):
        raise ValueError(
            "Eq. (9) is singular when a reference angle is near 90 degrees"
        )
    return np.cos(np.deg2rad(reference + deviation)) / denominator


def warp_doppler_time_map(
    dtm: np.ndarray,
    doppler_axis: ArrayLike,
    vector: ArrayLike,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Apply the transformation vector to a DTM's Doppler axis.

    ``dtm`` must be shaped ``[doppler_bins, frames]`` or
    ``[doppler_bins, frames, channels]``. A negative vector component mirrors
    the corresponding frame around zero Doppler, matching the sign reversal
    described for nonlinear gestures in the paper.
    """

    data = np.asarray(dtm)
    if data.ndim not in (2, 3):
        raise ValueError("dtm must have shape [doppler, time] or [doppler, time, channels]")
    axis = _as_1d_float("doppler_axis", doppler_axis)
    if axis.size != data.shape[0] or not np.all(np.diff(axis) > 0):
        raise ValueError("doppler_axis must be strictly increasing and match dtm rows")

    factors = np.asarray(vector, dtype=np.float64)
    if factors.ndim == 0:
        factors = np.full(data.shape[1], float(factors))
    if factors.ndim != 1 or factors.size != data.shape[1]:
        raise ValueError("vector must be scalar or contain one value per DTM frame")
    if not np.all(np.isfinite(factors)):
        raise ValueError("vector contains NaN or infinite values")

    work = data[:, :, None] if data.ndim == 2 else data
    output = np.full(
        work.shape,
        fill_value,
        dtype=np.result_type(work.dtype, np.asarray(fill_value).dtype),
    )
    zero_bin = int(np.argmin(np.abs(axis)))

    for frame, factor in enumerate(factors):
        if np.isclose(factor, 0.0, atol=1e-12):
            output[zero_bin, frame, :] = np.max(work[:, frame, :], axis=0)
            continue

        mapped_axis = axis * factor
        if factor < 0:
            mapped_axis = mapped_axis[::-1]
            values = work[::-1, frame, :]
        else:
            values = work[:, frame, :]
        boundary_tolerance = (
            np.finfo(np.float64).eps
            * max(1.0, abs(mapped_axis[0]), abs(mapped_axis[-1]))
            * 64
        )
        inside = (
            (axis >= mapped_axis[0] - boundary_tolerance)
            & (axis <= mapped_axis[-1] + boundary_tolerance)
        )
        query_axis = np.clip(axis[inside], mapped_axis[0], mapped_axis[-1])
        for channel in range(work.shape[2]):
            output[inside, frame, channel] = np.interp(
                query_axis,
                mapped_axis,
                values[:, channel],
            )

    return output[:, :, 0] if data.ndim == 2 else output
