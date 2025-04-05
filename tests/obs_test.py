import matplotlib.pyplot as plt
import numpy as np
import pytest
from rlgym.rocket_league.api import PhysicsObject

import env.denbot_obs as obs
import env.encoders as encoders


@pytest.fixture
def default_car():
    car = PhysicsObject()
    car.position = np.array([0, 0, 17])
    car.linear_velocity = np.array([1, 0, 0])
    car.angular_velocity = np.array([0, 0, 0])
    car._rotation_mtx = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return car


@pytest.mark.parametrize(
    "v, expected",
    [
        ((0, 1, 0), np.pi / 2),
        ((1, 0, 0), 0),
        ((0, 1, 0.5), np.pi / 2),
        ((1, 1, 0.5), np.pi / 4),
        ((-1, 1, 0), 3 * np.pi / 4),
        ((0, -1, 0), -np.pi / 2),
        ((-1, 0, 0), np.pi),
        ((np.sqrt(3) / 2, 0.5, 0), np.pi / 6),
        ((0, 1, 1), np.pi / 2),
        ((0, -1, 1), -np.pi / 2),
        ((1, 0, 0.1), 0),
        ((-1, 0, 0), np.pi),
        ((-1, 0, -1), np.pi),
        ((0, 0, -1), 0),
        ((0, 0, 1), 0),
    ],
)
def test_yaw_angles(default_car, v, expected):
    ref = default_car.forward
    norm = default_car.up
    assert np.isclose(encoders.planar_angle(ref, norm, np.array(v)), expected)


@pytest.mark.parametrize(
    "v, expected",
    [
        ((0, 1, 0), 0),
        ((1, 0, 0), 0),
        ((0, 1, 1), np.pi / 2),
        ((1, 1, 1), np.pi / 4),
        ((-1, 0, 1), 3 * np.pi / 4),
        ((0, 0, -1), -np.pi / 2),
        ((-1, 0, 0), np.pi),
        ((np.sqrt(3) / 2, 0, 0.5), np.pi / 6),
        ((0, 1, 1), np.pi / 2),
        ((1, 0.1, 0), 0),
    ],
)
def test_pitch_angles(default_car, v, expected):
    ref = default_car.forward
    norm = default_car.left
    assert np.isclose(encoders.planar_angle(ref, norm, np.array(v)), expected)


def test_fourier_encoder():
    x_min, x_max = -np.pi / 2, np.pi / 2
    x = np.linspace(-np.pi / 2, np.pi / 2, 5000)
    feats = obs.fourier_encoder(x_min, x_max, x, 3, periodic=False)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x, feats)
    plt.show()


def test_binary_encoding():
    assert np.all(np.array([1, 1, 1, 0, 0]) == obs.binary_encoder(10, 10 + 31, 10 + 7, num_bins=5))
