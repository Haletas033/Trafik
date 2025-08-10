import skimage.measure
import svgwrite
import os
import numpy as np
from scour import scour

from scipy.ndimage import center_of_mass, shift, gaussian_filter1d


def smooth_contour(contour, sigma=2):
    xs = gaussian_filter1d(contour[:, 1], sigma)
    ys = gaussian_filter1d(contour[:, 0], sigma)
    return np.vstack((ys, xs)).T


def create_image(zone_masks, colours, roads, buildings, shape, building_size=2):
    combined_mask = np.any(zone_masks, axis=0)

    cy, cx = center_of_mass(combined_mask)

    center_y, center_x = shape[0] / 2, shape[1] / 2

    shift_y = center_y - cy
    shift_x = center_x - cx

    dwg = svgwrite.Drawing("temp_city.svg", size=(shape[1], shape[0]))

    for zone_mask, color in zip(zone_masks, colours):
        shifted_mask = shift(zone_mask.astype(float), shift=(shift_y, shift_x), order=0, mode='constant', cval=0)
        shifted_mask = shifted_mask > 0.5

        contours = skimage.measure.find_contours(shifted_mask, 0.5)

        for contour in contours:
            smoothed = smooth_contour(contour, sigma=2)
            points = [(x, y) for y, x in smoothed]
            dwg.add(dwg.polygon(points, fill=color))

    for (x1, y1), (x2, y2) in roads:
        sx1, sy1 = x1 + shift_x, y1 + shift_y
        sx2, sy2 = x2 + shift_x, y2 + shift_y
        dwg.add(dwg.line(start=(sx1, sy1), end=(sx2, sy2), stroke='black', stroke_width=1.5))

    for building_corners in buildings:
        print(building_corners)
        shifted_points = [(x + shift_x, y + shift_y) for x, y in building_corners]
        dwg.add(dwg.polygon(shifted_points, fill='gray', stroke='black', stroke_width=0.5))

    dwg.save()

def optimize_svg(output_file):
    options = scour.sanitizeOptions()
    options.remove_metadata = True
    options.enable_viewboxing = True
    options.strip_comments = True
    options.shorten_ids = True
    options.float_precision = 2

    with open("temp_city.svg", "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        svg_data = infile.read()
        optimized_svg = scour.scourString(svg_data, options)
        outfile.write(optimized_svg)
    os.remove("temp_city.svg")