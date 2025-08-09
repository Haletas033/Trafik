import random
import numpy as np

def create_zones(mask, world, thresholds=None, colours=None):
    if thresholds is None:
        thresholds = [0.3, 0.6]
    if colours is None:
        colours = []
        for _ in thresholds:
            random_integer = random.randint(0, 0xFFFFFF)
            hex_color = '#{:06x}'.format(random_integer)

            colours.append(hex_color)


    previous_threshold = 0.0
    zone_masks = []

    extended_thresholds = thresholds + [1.1]

    for threshold in extended_thresholds:
        # Mask for pixels inside `mask` AND between previous and current threshold
        zone_mask = (world >= previous_threshold) & (world < threshold) & mask
        zone_masks.append(zone_mask)
        previous_threshold = threshold

    return zone_masks, colours