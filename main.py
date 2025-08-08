import noise
import numpy as np
import skimage.measure
import svgwrite
from scipy.ndimage import label, binary_fill_holes, center_of_mass, shift, gaussian_filter1d


def create_noise(shape, scale, octaves, persistence, lacunarity, seed):
    # Coordinate grid
    x_idx = np.linspace(0, 1, shape[0])
    y_idx = np.linspace(0, 1, shape[1])
    world_x, world_y = np.meshgrid(x_idx, y_idx)

    # Generate noise
    world = np.vectorize(noise.pnoise2)(
        world_x / scale,
        world_y / scale,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
        repeatx=1024,
        repeaty=1024,
        base=seed
    )

    # Normalize to 0–1 range
    world = (world - world.min()) / (world.max() - world.min())
    return world

def calculate_distance_to_edge(shape):
    # Pixel grid for distance calculation
    pixel_x = np.arange(shape[0])
    pixel_y = np.arange(shape[1])
    px_grid_x, px_grid_y = np.meshgrid(pixel_x, pixel_y)

    # Distance to nearest edge (in pixels)
    dist_left = px_grid_x
    dist_right = shape[0] - 1 - px_grid_x
    dist_top = px_grid_y
    dist_bottom = shape[1] - 1 - px_grid_y

    dist_to_edge = np.minimum(np.minimum(dist_left, dist_right), np.minimum(dist_top, dist_bottom))
    return dist_to_edge

def calculate_threshold(shape, dist_to_edge, center_thresh, floor_thresh, min_thresh, edge_band):
    # Calculate linear threshold from center to edges (0 at edges)
    max_dist = shape[0] // 2  # max distance approx from center to edge (512)
    threshold = center_thresh * (dist_to_edge / max_dist)

    # Clamp threshold to floor_thresh in last 10 pixels near edges
    threshold = np.where(dist_to_edge < edge_band, floor_thresh, threshold)

    # Clamp threshold to valid range
    threshold = np.clip(threshold, min_thresh, center_thresh)
    return threshold

def remove_islands(mask):
    # Label connected components (8-connectivity)
    labeled_array, num_features = label(mask)

    # Find the largest connected component
    if num_features > 0:
        # Count pixels in each component
        counts = np.bincount(labeled_array.ravel())
        counts[0] = 0  # background label 0 ignored
        largest_label = counts.argmax()

        # Create a mask keeping only the largest connected component
        mask = (labeled_array == largest_label)

    # Fill any holes in the main landmass
    country_mask_filled = binary_fill_holes(mask)
    return country_mask_filled

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

# Parameters
SHAPE = (1024, 1024)
SCALE = 0.5
OCTAVES = 6
PERSISTENCE = 0.5
LACUNARITY = 2.0
SEED = np.random.randint(0, 100)

EDGE_BAND = 10
CENTER_THRESH = 0.7
MIN_THRESH = 0.0
FLOOR_THRESH = 0.0

UPSCALE_FACTOR = 4
HIGH_RES_SHAPE = (SHAPE[0] * UPSCALE_FACTOR, SHAPE[1] * UPSCALE_FACTOR)

#---Generate the country---#

created_noise = create_noise(SHAPE, SCALE, OCTAVES, PERSISTENCE, LACUNARITY, SEED)

calculated_threshold = calculate_threshold(SHAPE, calculate_distance_to_edge(SHAPE),
                                           CENTER_THRESH, FLOOR_THRESH, MIN_THRESH, EDGE_BAND)

# Create mask for country (white land)
country_mask = created_noise < calculated_threshold

# Remove islands and create the image
create_image(remove_islands(country_mask), SHAPE)