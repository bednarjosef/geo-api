import numpy as np


def xyz_to_latlon(x, y, z):
    lats = np.degrees(np.arcsin(z))
    lons = np.degrees(np.arctan2(y, x))
    return lats, lons
