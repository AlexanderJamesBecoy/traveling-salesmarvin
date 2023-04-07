from svg.path import parse_path
from xml.dom import minidom
import numpy as np


def get_point_at(path, distance, scale, offset):
    pos = path.point(distance)
    pos += offset
    pos *= scale
    return pos.real, pos.imag

def points_from_path(path, density, scale, offset):
    step = int(path.length() * density)
    last_step = step - 1

    if last_step == 0:
        yield get_point_at(path, 0, scale, offset)
        return

    for distance in range(step):
        yield get_point_at(
            path, distance / last_step, scale, offset)


def points_from_doc(doc, density=5, scale=1, offset=0):
    offset = offset[0] + offset[1] * 1j
    points = []
    for element in doc.getElementsByTagName("path"):
        for path in parse_path(element.getAttribute("d")):
            points.extend(points_from_path(
                path, density, scale, offset))

    return points

def get_points(filename, density=0.05, scale=1):
    doc = minidom.parse(filename)
    points = points_from_doc(doc, density, scale, (0, 0))
    doc.unlink()
    return points

def center_logo(points, offset=[0.,0.]):
    mean = np.mean(points, axis=0) + np.array(offset)
    centered_points = points - np.tile(mean.reshape(1,-1), (len(points),1))
    return centered_points

def orientate_logo(points, theta=np.pi):
    R1 = [[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]
    R2 = [[-1,0],[0,1]]
    points = np.matmul(R2, np.matmul(R1, np.array(points).T)).T
    return points

def generate_noise(points, var=1.0):
    new_points = np.zeros(points.shape)
    for idx, point in enumerate(points):
        new_points[idx][0] = np.random.normal(point[0], var)
        new_points[idx][1] = np.random.normal(point[1], var)
    return new_points

def import_pmb(filename, image_height, image_width, density):
    points = []

    with open(filename, "r") as f:
        for line in f.readlines()[3:]:
            for bit in line:
                if bit != '\n':
                    points.append(int(bit))

    points = np.array(points).reshape(image_height, image_height)
    points = np.vstack(np.where(points == 1)).T
    points = orientate_logo(points, 1.5*np.pi)
    points = (np.array([[-1.,0.],[0.,1.]]) @ points.T).T
    density = int(1.0/density)

    return points[::density]