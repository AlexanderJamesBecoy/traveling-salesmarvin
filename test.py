import gmplot
import numpy as np
import matplotlib.pyplot as plt
import tsp, logo, time

"""
Credits:
- svg path parser: https://stackoverflow.com/questions/69313876/how-to-get-points-of-the-svg-paths
- gmplot: https://www.javatpoint.com/plotting-google-map-using-gmplot-package-in-python
- TSP/ILP: RO47005 - Planning & Decision Making Lecture 16
"""

points = logo.get_points(density=0.01)
points = logo.orientate_logo(points)
points = logo.generate_noise(points, var=3.0)
print("Number of points: {}".format(len(points)))

plt.figure()
plt.scatter(points[:,0], points[:,1])
plt.show()

start_time = time.time()
cost_matrix = tsp.def_cost_matrix(points)
sol = tsp.solve_tsp(points, cost_matrix)
print("Vehicle follows the route: {}".format(sol))
time_diff = (time.time() - start_time)/60.0
print("Execution time: {} m".format(time_diff))
sol = np.array(sol) - 1

plt.figure()
plt.scatter(points[:,0], points[:,1])
plt.plot(points[sol][:,0], points[sol][:,1])
plt.savefig('figure.png', transparent=True)
plt.show()

# firstGmap = gmplot.GoogleMapPlotter(28.612894, 77.229446, 18)
# firstGmap.apikey = "AIzaSyDeRNMnZ__VnQDiATinz4kPjF_c9r1kWe8"  
# firstGmap.draw( "firstmap.html" )  