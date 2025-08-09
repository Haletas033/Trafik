import numpy as np

def create_zones(mask, world, thresholds=None, colours=None):
    if thresholds is None:
        thresholds = [0.3, 0.6]
    if colours is None:
        colours = ['#a6cee3', '#1f78b4', '#b2df8a']

    previous_threshold = 0.0
    zone_masks = []

    extended_thresholds = thresholds + [1.1]

    for threshold in extended_thresholds:
        # Mask for pixels inside `mask` AND between previous and current threshold
        zone_mask = (world >= previous_threshold) & (world < threshold) & mask
        zone_masks.append(zone_mask)
        previous_threshold = threshold

    return zone_masks, colours