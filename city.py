import noise
import numpy as np

from scipy.ndimage import label, binary_fill_holes

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
    city_mask_filled = binary_fill_holes(mask)
    return city_mask_filled