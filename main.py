import visualize as vis
import city
import zones
import roads
import buildings as bds

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

city_mask = city.remove_islands(city_mask)

#---Generate city zones---#
zone_thresholds = [0.3, 0.4, 0.6, 0.9]
zone_masks, colours = zones.create_zones(city_mask, created_noise, thresholds=zone_thresholds)

#---Generate roads---#

coords = roads.create_city_centers(city_mask, probability=0.0001)
road_paths = roads.generate_roads(city_mask, coords, step_length=10, split_prob=0.01)

#---Generate buildings---#

buildings, building_polygons = bds.add_buildings_along_roads_rotated(road_paths)
vis.create_image(zone_masks, colours, road_paths, building_polygons, SHAPE)


#---Visualize the city---#

# Remove islands and create the image
vis.create_image(zone_masks, colours, road_paths, building_polygons, SHAPE)
vis.optimize_svg("city.svg")
