import math
import random

import numpy as np


def create_city_centers(mask, probability=0.0001):
    city_centers = []

    valid_positions = np.argwhere(mask)

    selected = valid_positions[np.random.rand(len(valid_positions)) < probability]

    city_centers.extend([(col, row) for row, col in selected])
    return city_centers


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def lines_intersect(p1, p2, p3, p4):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def generate_roads(mask, city_centers, step_length=15, death_prob=0.02, snap_distance=10, split_prob=0.05, max_walkers=300):
    class Walker:
        def __init__(self, position, angle, is_original=False):
            self.position = position
            self.angle = angle
            self.alive = True
            self.path = [position]
            self.is_original = is_original


        def next_position(self):
            x, y = self.position
            new_x = x + step_length * math.cos(math.radians(self.angle))
            new_y = y + step_length * math.sin(math.radians(self.angle))
            return new_x, new_y


        def move(self, nodes):
            if not self.alive:
                return None, None

            steps_taken = len(self.path) - 1

            if (not self.is_original) and (steps_taken <= 3):
                if random.random() < death_prob:
                    self.alive = False
                    return None, None

            new_pos = self.next_position()

            for node_pos in nodes:
                if distance(new_pos, node_pos) < snap_distance:
                    return node_pos, True

            return new_pos, False


    def check_intersection(new_segment, segments):
        p1, p2 = new_segment
        for seg in segments:
            q1, q2 = seg
            if p1 == q1 or p1 == q2 or p2 == q1 or p2 == q2:
                continue
            if lines_intersect(p1, p2, q1, q2):
                return True
        return False

    def is_inside_mask(pos):
        x, y = pos

        x_int, y_int = int(round(x)), int(round(y))

        if 0 <= y_int < mask.shape[0] and 0 <= x_int < mask.shape[1]:
            return mask[y_int][x_int]
        else:
            return False


    def main():
        crosses = [(int(x), int(y)) for x, y in city_centers]
        walkers = []
        for cx, cy in crosses:
            for angle in [0, 90, 180, 270]:
                walkers.append(Walker((cx, cy), angle, is_original=True))

        segments = []
        nodes = set(crosses)

        while any(w.alive for w in walkers):
            new_walkers = []
            for w in walkers:
                if not w.alive:
                    continue

                if not is_inside_mask(w.position):
                    w.alive = False
                    continue

                new_pos, snapped = w.move(nodes)
                if new_pos is None:
                    continue

                new_segment = (w.position, new_pos)
                if check_intersection(new_segment, segments):
                    w.alive = False
                    continue

                if not snapped:
                    nodes.add(new_pos)

                segments.append(new_segment)

                w.position = new_pos
                w.path.append(new_pos)

                if random.random() < split_prob and len(walkers) + len(new_walkers) < max_walkers:
                    split_angle = w.angle + random.choice([90, -90])
                    new_walkers.append(Walker(new_pos, split_angle))

                if random.random() < 0.2:
                    w.angle += random.choice([-90, 90])

            walkers.extend(new_walkers)
        return segments

    return main()