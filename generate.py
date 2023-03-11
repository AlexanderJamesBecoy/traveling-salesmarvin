import numpy as np
import matplotlib.pyplot as plt
import tsp, logo, time
from concorde.tsp import TSPSolver

points = logo.get_points(density=0.13)
points = logo.orientate_logo(points)
points = logo.generate_noise(points, var=3.0)
np.random.shuffle(points)
print("Number of points: {}".format(len(points)))

plt.figure()
plt.scatter(points[:,0], points[:,1], s=0.5, color='black')
plt.savefig('figure_points.png', transparent=True)
plt.show()

solver = TSPSolver.from_data(points[:,0], points[:,1], norm="EUC_2D")
solution = solver.solve()
assert solution.success

path = points[solution.tour]
plt.figure()
plt.plot(path[:,0], path[:,1], linewidth=1.0, color='orange')
plt.savefig('figure_lines.png', transparent=True)
plt.show()