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

def get_points(density=0.05, scale=1):
    doc = minidom.parse('RSA_logo.svg')
    points = points_from_doc(doc, density, scale, (0, 0))
    doc.unlink()
    return points

def orientate_logo(points, theta=np.pi):
    R1 = [[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]
    R2 = [[-1,0],[0,1]]
    points = np.matmul(R2, np.matmul(R1, np.array(points).T)).T
    return points

def generate_noise(points):
    pass