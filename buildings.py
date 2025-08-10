import math

import numpy as np
from shapely import Point
from shapely.geometry import Polygon, LineString

def rotate_point(px, py, cx, cy, angle_rad):
    s, c = np.sin(angle_rad), np.cos(angle_rad)
    px -= cx
    py -= cy
    xnew = px * c - py * s
    ynew = px * s + py * c
    return xnew + cx, ynew + cy


def get_building_corners(center, size, angle_rad):
    half_w, half_h = size[0]/2, size[1]/2
    cx, cy = center

    # Rectangle corners before rotation, relative to center
    corners = np.array([
        [-half_w, -half_h],
        [half_w, -half_h],
        [half_w, half_h],
        [-half_w, half_h]
    ])

    rotated_corners = []
    for (x, y) in corners:
        rx, ry = rotate_point(cx + x, cy + y, cx, cy, angle_rad)
        rotated_corners.append([rx, ry])

    return np.array(rotated_corners)


from shapely.geometry import MultiLineString

def add_buildings_along_roads_rotated(segments, building_size=(6,4), building_spacing=7, max_offset=10, offset_step=0.5):
    buildings = []
    building_polygons = []

    half_diag = np.linalg.norm(building_size)/2
    min_distance = half_diag * 2

    # Combine all road segments into one MultiLineString for global distance checks
    all_roads_line = MultiLineString([LineString([p1, p2]) for p1, p2 in segments])

    def too_close(new_center):
        for poly in building_polygons:
            if Polygon(poly).distance(Point(new_center)) < min_distance:
                return True
        return False

    for p1, p2 in segments:
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        direction = p2 - p1

        length = np.linalg.norm(direction)
        if length == 0:
            continue

        direction /= length

        angle_rad = math.atan2(direction[1], direction[0])
        perp = np.array([-direction[1], direction[0]])

        num_buildings = int(length // building_spacing)

        for i in range(1, num_buildings + 1):
            base_pos = p1 + direction * i * building_spacing

            placed = False
            for side in [+1, -1]:
                offset = 0
                while offset <= max_offset:
                    candidate_center = base_pos + perp * side * offset
                    building_corners = get_building_corners(candidate_center, building_size, angle_rad)
                    building_poly = Polygon(building_corners)

                    # Check distance to *all* roads
                    if building_poly.distance(all_roads_line) > 0 and not too_close(candidate_center):
                        buildings.append(candidate_center)
                        building_polygons.append(building_corners)
                        placed = True
                        break
                    offset += offset_step
                if placed:
                    break

    return buildings, building_polygons
