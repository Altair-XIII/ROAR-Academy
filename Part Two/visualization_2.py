import matplotlib.pyplot as plt
import numpy as np

# Data points
x = [0.0, 2.0, 3.0]  # x coordinates
y = [2.0, 4.0, 0.0]  # y coordinates

plt.plot(x, y, color='blue')

plt.xlim(0.0, 3.0)
plt.ylim(0.0, 4.0)

plt.xticks(np.arange(0, 3.5, 0.5))
plt.yticks(np.arange(0, 4.5, 0.5))

plt.title("Sample graph!")
plt.xlabel("x - axis")
plt.ylabel("y - axis")

plt.show()