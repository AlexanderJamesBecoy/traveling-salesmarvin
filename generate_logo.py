from logo import import_pmb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from concorde.tsp import TSPSolver

if __name__ == "__main__":

    # Generate array of points from the PMB file
    points = import_pmb("RSA_logo_stippled.pbm", 480, 480, 0.1)
    print(points.shape)
    plt.figure()
    plt.scatter(points[:,0], points[:,1], s=0.05, color='black', alpha=1.0)
    plt.show()

    # Cluster the regions of points
    own_colors = ["blue", "green", "yellow", "purple", "black", "lime", "orange", "cyan", "gray", "crimson", "gold", "pink"]
    dbs = DBSCAN(eps=20, min_samples=3)
    labels = dbs.fit_predict(points)
    print(len(np.unique(labels[labels != -1])))
    plt.figure(figsize=(8,6))
    for i in range(len(points)):
        plt.scatter(points[i,0], points[i,1], s=0.5, marker='.', color=own_colors[labels[i]])
    # plt.savefig('figure_points.png', transparent=True)
    plt.show()

    # Apply TSP to each cluster
    label_classes = np.unique(labels)
    plt.figure(figsize=(8,6))
    for label_class in label_classes:
        points_label = points[labels == label_class]
        if len(points_label) <= 1:
            continue
        # points_label = points_label[::2]
        solver = TSPSolver.from_data(points_label[:,0], points_label[:,1], norm="EUC_2D", name='RSA_logo')
        solution = solver.solve(verbose=False)
        assert solution.success

        path = points_label[solution.tour]
        plt.plot(path[:,0], path[:,1], linewidth=0.5, color='black')#own_colors[label_class])
    plt.savefig('figure_lines.png', transparent=True)
    plt.show()

