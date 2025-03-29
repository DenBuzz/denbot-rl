import numpy as np
import pytest
from rlgym.rocket_league.api import PhysicsObject

from env.obs_builder import planar_angle


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
    assert np.isclose(planar_angle(ref, norm, np.array(v)), expected)


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
    assert np.isclose(planar_angle(ref, norm, np.array(v)), expected)


# def test_planar():
#     r = np.array([-0.6537304, 0.7480598, 0.1142068])
#     n = np.array([-0.7456911, -0.6624989, 0.07099336])
#     t = np.array([-2030.0193, -5929.5356, 31.78704])
#
#     assert planar_angle(r, n, t) > 100
