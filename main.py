import visualize as vis
import city

import numpy as np

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

#---Generate the city---#

created_noise = city.create_noise(SHAPE, SCALE, OCTAVES, PERSISTENCE, LACUNARITY, SEED)

calculated_threshold = city.calculate_threshold(SHAPE, city.calculate_distance_to_edge(SHAPE),
                                           CENTER_THRESH, FLOOR_THRESH, MIN_THRESH, EDGE_BAND)

# Create mask for city (white land)
city_mask = created_noise < calculated_threshold

#---Visualize the city---#

# Remove islands and create the image
vis.create_image(city.remove_islands(city_mask), SHAPE)