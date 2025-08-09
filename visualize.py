import skimage.measure
import svgwrite
import numpy as np

from scipy.ndimage import center_of_mass, shift, gaussian_filter1d


def smooth_contour(contour, sigma=2):
    xs = gaussian_filter1d(contour[:, 1], sigma)
    ys = gaussian_filter1d(contour[:, 0], sigma)
    return np.vstack((ys, xs)).T


def create_image(zone_masks, colours, shape):
    combined_mask = np.any(zone_masks, axis=0)

    cy, cx = center_of_mass(combined_mask)

    center_y, center_x = shape[0] / 2, shape[1] / 2

    shift_y = center_y - cy
    shift_x = center_x - cx

    dwg = svgwrite.Drawing("city.svg", size=(shape[1], shape[0]))

    for zone_mask, color in zip(zone_masks, colours):
        shifted_mask = shift(zone_mask.astype(float), shift=(shift_y, shift_x), order=0, mode='constant', cval=0)
        shifted_mask = shifted_mask > 0.5

        contours = skimage.measure.find_contours(shifted_mask, 0.5)

        for contour in contours:
            smoothed = smooth_contour(contour, sigma=2)
            points = [(x, y) for y, x in smoothed]
            dwg.add(dwg.polygon(points, fill=color))

    dwg.save()
