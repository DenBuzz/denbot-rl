import numpy as np
from numpy.linalg import norm
from rlgym.rocket_league.common_values import BACK_NET_Y, CEILING_Z, SIDE_WALL_X


def encode_position(position: np.ndarray, frequencies: int = 4) -> np.ndarray:
    """Encode the shit out of some positions"""
    encoded = [
        fourier_encoder(-SIDE_WALL_X, SIDE_WALL_X, position[0], frequencies),
        fourier_encoder(-BACK_NET_Y, BACK_NET_Y, position[1], frequencies),
        fourier_encoder(-CEILING_Z, CEILING_Z, position[2], frequencies),
    ]
    return np.concatenate(encoded)


def encode_velocity(max_speed: float, vel: np.ndarray, frequencies: int = 4) -> np.ndarray:
    """Encode the shit out of some positions"""
    encoded = [
        fourier_encoder(-max_speed, max_speed, vel, frequencies),
    ]
    return np.concatenate(encoded)


def fourier_encoder(low, high, value: float | np.ndarray, frequencies=4, periodic=False):
    n_range = np.arange(frequencies)
    if periodic:
        n_range += 1
    freqs = np.exp2(n_range)

    value = np.array(value).reshape(-1, 1)
    trig_params = (value - (low + high) / 2) * freqs * (2 * np.pi / (2 * (high - low)))
    sin_terms = np.sin(trig_params)
    cos_terms = np.cos(trig_params)
    return np.concatenate((sin_terms, cos_terms), axis=-1).squeeze()


def binary_encoder(low, high, value, num_bins: int) -> np.ndarray:
    value = np.clip(value, low, high)
    mask = 2 ** np.arange(num_bins)
    max = 2**num_bins - 1

    scaled_val = int((value - low) / (high - low) * max)
    encoding = scaled_val & mask != 0
    return encoding.astype(float)


def planar_angle(reference: np.ndarray, normal: np.ndarray, target: np.ndarray) -> float:
    """Return the yaw angle between a car's foward and a unit vector"""
    if (x := norm(normal)) == 0:
        return 0
    else:
        n = normal / x

    t_proj = target - np.dot(target, n) * n
    if (x := norm(t_proj)) == 0:
        return 0
    else:
        t_proj_norm = t_proj / x

    ref_proj = reference - np.dot(reference, n) * n
    if (x := norm(ref_proj)) == 0:
        return 0
    else:
        ref_proj_norm = ref_proj / x

    angle = np.arccos(np.clip(np.dot(ref_proj_norm, t_proj_norm), -1, 1))
    sign = np.sign(np.dot(np.cross(ref_proj_norm, t_proj_norm), n))
    if sign == 0:
        return angle
    return angle * sign
