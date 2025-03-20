import numpy as np
import matplotlib.pyplot as plt


def fourier_encoder(min, max, value, n=4):
    freqs = np.exp2(np.arange(n))
    return np.sin((value - (min + max) / 2) * freqs * (2 * np.pi / (2 * (max - min))))


def test_fourier_encoder():
    x_min, x_max = 0, 5120
    x = np.linspace(x_min, x_max, 10000)
    feats = np.array([fourier_encoder(x_min, x_max, v, 6) for v in x])

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x, feats)
    plt.show()


if __name__ == "__main__":
    test_fourier_encoder()
