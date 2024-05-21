
from .cellspace import CellSpace
from models.utils import meters2lonlat, lonlat2meters

def init_cs(min_lon, min_lat, max_lon, max_lat, cellspace_buffer, cell_size):
    #print("Init cellspace")
    x_min, y_min = lonlat2meters(min_lon, min_lat)
    x_max, y_max = lonlat2meters(max_lon, max_lat)
    x_min -= cellspace_buffer
    y_min -= cellspace_buffer
    x_max += cellspace_buffer
    y_max += cellspace_buffer

    cell_size = int(cell_size)
    cs = CellSpace(cell_size, cell_size, x_min, y_min, x_max, y_max)
    return cs