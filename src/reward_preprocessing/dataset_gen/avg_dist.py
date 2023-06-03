import matplotlib
import matplotlib.pyplot as plt
import numpy as np

size = (2, 2)  # In inches
number_pairs = 3
circle_radius = 0.05
colors = ["r", "b", "g"]  # Add more colores to allow more pairs
matplotlib.use("TkAgg")
seed = 0

# Set the seed
np.random.seed(seed)

if number_pairs > len(colors):
    raise ValueError("Not enough colors for the number of pairs")

fig, ax = plt.subplots()
fig.set_size_inches(size)

distances = []  # Collect distances between same-colored circles
for pair_i in range(number_pairs):
    a_x = np.random.uniform(0, 1)
    a_y = np.random.uniform(0, 1)
    a = plt.Circle(
        (a_x, a_y),
        circle_radius,
        color=colors[pair_i],
        clip_on=False,
    )

    ax.add_patch(a)

    b_x = np.random.uniform(0, 1)
    b_y = np.random.uniform(0, 1)
    b = plt.Circle(
        (b_x, b_y),
        circle_radius,
        color=colors[pair_i],
        clip_on=False,
    )

    ax.add_patch(b)

    distance = np.sqrt((a_x - b_x) ** 2 + (a_y - b_y) ** 2)
    distances.append(distance)
    
plt.axis("off")
fig.savefig("plotcircles.png", bbox_inches="tight")
