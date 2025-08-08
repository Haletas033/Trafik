import skimage.measure
import svgwrite
import numpy as np

from scipy.ndimage import center_of_mass, shift, gaussian_filter1d

def smooth_contour(contour, sigma=2):
    xs = gaussian_filter1d(contour[:, 1], sigma)
    ys = gaussian_filter1d(contour[:, 0], sigma)
    return np.vstack((ys, xs)).T

def create_image(filled_mask, shape):
    cy, cx = center_of_mass(filled_mask)

    center_y, center_x = shape[0] / 2, shape[1] / 2

    shift_y = center_y - cy
    shift_x = center_x - cx

    shifted_mask = shift(filled_mask.astype(float), shift=(shift_y, shift_x), order=0, mode='constant', cval=0)
    shifted_mask = shifted_mask > 0.5

    contours = skimage.measure.find_contours(shifted_mask, 0.5)

    dwg = svgwrite.Drawing("city.svg", size=(shape[1], shape[0]))
    for contour in contours:
        smoothed = smooth_contour(contour, sigma=2)
        points = [(x, y) for y, x in smoothed]
        dwg.add(dwg.polygon(points, fill='white', stroke='black', stroke_width=1))
    dwg.save()