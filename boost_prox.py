import matplotlib.pyplot as plt
import numpy as np
import rlgym.rocket_league.common_values as cv

timer_mask = np.random.random(34)
timer_mask = timer_mask > 0.1


def boost_prox_reward(pos: np.ndarray):
    """Return proximity to boosts"""
    boosts = np.array(cv.BOOST_LOCATIONS)

    big_pad_indices = [3, 4, 15, 18, 29, 30]
    pad_amounts = 12 * np.ones(boosts.shape[0])
    pad_amounts[big_pad_indices] = 100

    distances = np.linalg.norm(boosts[:, :2] - pos[:2], axis=1)
    rewards = np.sqrt(pad_amounts / 100) * np.exp(-0.002 * distances) # * timer_mask

    return np.sum(rewards)


x_range = (-cv.SIDE_WALL_X, cv.SIDE_WALL_X)  # Adjust as needed
y_range = (-cv.BACK_WALL_Y, cv.BACK_WALL_Y)  # Adjust as needed
resolution = 200  # Number of points along each axis

# Create a grid of car positions
x_values = np.linspace(*x_range, resolution)
y_values = np.linspace(*y_range, resolution)
X, Y = np.meshgrid(x_values, y_values)

reward_matrix = np.zeros((resolution, resolution))

for i in range(resolution):
    for j in range(resolution):
        car_position = np.array([X[i, j], Y[i, j]])
        reward_matrix[i, j] = boost_prox_reward(car_position)

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, reward_matrix, levels=100, cmap="viridis")
plt.colorbar(label="Reward")
plt.title("Reward Surface Based on Car Position Relative to Boost Pads")
plt.xlabel("X Position of Car")
plt.ylabel("Y Position of Car")
plt.axis("equal")
plt.grid(True)
plt.show()
