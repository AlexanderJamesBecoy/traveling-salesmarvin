import numpy as np
import matplotlib.pyplot as plt
import tsp, logo, time
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from concorde.tsp import TSPSolver

# points = logo.get_points('RSA_logo_delauney_v2.svg', density=0.1, scale=0.01)
points = logo.get_points('RSA_logo.svg', density=0.5, scale=1)#scale=0.01)
# points = logo.orientate_logo(points, theta=np.pi*1.15)
points = logo.orientate_logo(points)
points = logo.center_logo(points, offset=[0.,45.])
points = logo.generate_noise(points, var=0.5)
np.random.shuffle(points)
print("Number of points: {}".format(len(points)))

plt.figure()
plt.scatter(points[:,0], points[:,1], s=0.1, color='black')
plt.savefig('figure_points.png', transparent=True)
plt.show()

"""
Feature Extraction:
x = r cos t, y = r sin t
r = sqrt(x^2 + y^2), t = atan(y/x)
"""
# points_polar = np.zeros(points.shape)
# for i in range(len(points_polar)):
#     points_polar[i,0] = np.linalg.norm(points[i], ord=3)
#     points_polar[i,1] = np.abs(np.arctan2(points[i,1], points[i,0]))
# plt.figure()
# plt.scatter(points_polar[:,0], points_polar[:,1], s=0.5, color="black")
# plt.show()

n_clusters = 2
own_colors = ["blue", "green", "yellow", "purple", "black", "lime", "orange", "cyan", "gray", "crimson", "gold", "pink"]

# kmean = KMeans(n_clusters=n_clusters)
# labels = kmean.fit_predict(points_polar)
# gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag')
# labels = gmm.fit_predict(points_polar)
dbs = DBSCAN(eps=10, min_samples=8)
labels = dbs.fit_predict(points)

plt.figure()
for i in range(len(points)):
    plt.scatter(points[i,0], points[i,1], s=0.5, color=own_colors[labels[i]])
# plt.savefig('figure_points.png', transparent=True)
plt.show()

## TSP solver
# solver = TSPSolver.from_data(points[:,0], points[:,1], norm="EUC_2D", name='RSA_logo')
# solution = solver.solve(verbose=False)
# assert solution.success

# path = points[solution.tour]
# plt.figure()
# plt.plot(path[:,0], path[:,1], linewidth=0.8, color='orange')
# plt.savefig('figure_lines.png', transparent=True)
# plt.show()

plt.figure()
label_classes = np.unique(labels)

for label_class in label_classes:
    points_label = points[labels == label_class]
    points_label = points_label[::2]
    solver = TSPSolver.from_data(points_label[:,0], points_label[:,1], norm="EUC_2D", name='RSA_logo')
    solution = solver.solve(verbose=False)
    assert solution.success

    path = points_label[solution.tour]
    plt.plot(path[:,0], path[:,1], linewidth=0.8, color=own_colors[label_class])
plt.savefig('figure_lines.png', transparent=True)
plt.show()