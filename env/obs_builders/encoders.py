import numpy as np
import matplotlib.pyplot as plt
from rlgym.rocket_league.common_values import SIDE_WALL_X, CEILING_Z, BACK_NET_Y


def encode_position(position: np.ndarray, frequencies: int = 4) -> np.ndarray:
    """Encode the shit out of some positions"""
    encoded = [
        fourier_encoder(-SIDE_WALL_X, SIDE_WALL_X, position[0], frequencies),
        fourier_encoder(-BACK_NET_Y, BACK_NET_Y, position[1], frequencies),
        fourier_encoder(-CEILING_Z, CEILING_Z, position[2], frequencies),
    ]
    return np.concatenate(encoded)


def fourier_encoder(min, max, value, frequencies=4, periodic=False):
    n_range = np.arange(frequencies)
    if periodic:
        n_range += 1
    freqs = np.exp2(n_range)
    trig_params = (value - (min + max) / 2) * freqs * (2 * np.pi / (2 * (max - min)))
    sin_terms = np.sin(trig_params)
    cos_terms = np.cos(trig_params)
    return np.concatenate([sin_terms, cos_terms])


def test_fourier_encoder():
    x_min, x_max = -1000, 2000
    x = np.linspace(x_min, x_max, 10_000)
    feats = np.array([fourier_encoder(x_min, x_max, v, 6, periodic=False) for v in x])

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x, feats)
    plt.show()


if __name__ == "__main__":
    test_fourier_encoder()
