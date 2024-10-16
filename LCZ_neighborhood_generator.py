# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:47:02 2024

@author: Jannik Heusinger

This script creates a plan view of a city district according to LCZ parameters. It can be used to
create neighborhoods (with simplified building blocks similar e.g. to the superblocks in Barcelona) 
in accordance to specific LCZ types and the exported image can be used as input to Envi-met.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.ndimage as ndimage
from tqdm import tqdm

lcz_params = {
    "LCZ1": {"height_range": (25, 100), "density": 0.5, "street_width_range": (15, 15)},  # Compact high-rise, narrow streets
    "LCZ2": {"height_range": (10, 25), "density": 0.55, "street_width_range": (15, 15)},  # Compact mid-rise
    "LCZ3": {"height_range": (3, 10), "density": 0.55, "street_width_range": (8, 8)},  # Compact low-rise
    "LCZ4": {"height_range": (25, 100), "density": 0.3, "street_width_range": (20, 20)}, # Open high-rise
    "LCZ5": {"height_range": (10, 25), "density": 0.3, "street_width_range": (20, 20)},  # Open mid-rise
    "LCZ6": {"height_range": (3, 10), "density": 0.3, "street_width_range": (15, 15)},  # Open low-rise, suburban areas
}

#### User Input here ####
size_m = 1000           # model area size in x and y direction (m)
resolution = 1          # horizontal resolution (m)
lcz_type="LCZ2"         # local climate zone type
road_spacing_m = 82     # distance between roads (center point) - this essentially defines the city block sizes (m)

road_spacing = road_spacing_m/resolution # distance between roads (grids)
size=size_m/resolution                  # model area size in x and y direction (grids)
def generate_full_road_network_with_width(size, lcz_type, lcz_params, road_spacing):
    """
    Generate a road network with variable street widths based on the LCZ type.
    
    Args:
    - size (int): The size of the grid (in meters, where 1 cell = 1 meter).
    - lcz_type (str): The LCZ type that defines the street width.
    - lcz_params (dict): Parameters for the LCZ, including street width ranges.
    - road_spacing (int): Spacing between roads in grid cells.
    
    Returns:
    - road_grid (np.array): A grid with roads (1) and empty space (0).
    """
    road_grid = np.zeros((size, size))
    
    # Get the street width range for the current LCZ
    street_width_min, street_width_max = lcz_params[lcz_type]["street_width_range"]
    
    for i in range(10, size-10, road_spacing):
        # Randomly pick a street width within the LCZ's street width range
        street_width = random.randint(street_width_min, street_width_max)
        
        # Horizontal roads
        road_grid[i:i+street_width, :] = 1
        
        # Vertical roads
        road_grid[:, i:i+street_width] = 1
    
    return road_grid

def place_buildings_along_roads(road_grid, lcz_params, lcz_type ,building_width=10):
    """
    Places buildings along roads, ensuring a continuous row of buildings adjacent to streets
    and avoiding placing buildings inside the roads.
    
    Args:
    - road_grid (np.array): A 2D array with roads (1) and empty space (0).
    - lcz_params (dict): Parameters for the LCZ, including height and density.
    - lcz_type (str): The LCZ type being used for the neighborhood.
    
    Returns:
    - building_grid (np.array): A grid with building footprints.
    """
    max_building_height = lcz_params["LCZ2"]["height_range"][1]
    building_grid = np.zeros_like(road_grid)
    
    for i in range(10, road_grid.shape[0] - 10):  # Exclude boundary rows
        for j in range(10, road_grid.shape[1] - 10):  # Exclude boundary columns
            # Check if the current cell is empty and next to a road
            if road_grid[i, j] == 0 and (
                road_grid[i-1, j] == 1 or road_grid[i+1, j] == 1 or  # Check above or below
                road_grid[i, j-1] == 1 or road_grid[i, j+1] == 1):   # Check left or right
                
                
                if (road_grid[i-1,1] == 1) & (i + building_width < road_grid.shape[0]):
                    building_grid[i:i + building_width, j] = 1
                if (road_grid[i+1, j] == 1) & (i - building_width > 0):
                    building_grid[i - building_width:i, j] = 1
                if road_grid[i, j-1] == 1 & (j + building_width < road_grid.shape[1]):
                    building_grid[i,j:j+building_width] = 1
                if road_grid[i, j+1] == 1 & (j - building_width > 0):
                    building_grid[i, j - building_width:j] = 1
                
                # buildings near the border are not allowed in Envi-met
                # therefore a horizontal distance = max building_height will be set to 0
                building_grid[0:max_building_height,:] = 0
                building_grid[-1-max_building_height:,:] = 0
                building_grid[:,0:max_building_height] = 0
                building_grid[:,-1-max_building_height:] = 0
    
    return building_grid

def assign_uniform_building_heights_with_density(building_grid, lcz_params, lcz_type):
    """
    Assigns a uniform building height to each enclosed block of buildings.
    
    Args:
    - building_grid (np.array): A grid with buildings marked as 1.
    - lcz_params (dict): Parameters for the LCZ, including height range.
    - lcz_type (str): The LCZ type being used for the neighborhood.
    
    Returns:
    - height_grid (np.array): A grid with building heights assigned to each block.
    """
    height_range = lcz_params[lcz_type]["height_range"]
    
    # Label each separate building block
    labeled_blocks, num_blocks = ndimage.label(building_grid)
    
    # Create a grid to store building heights
    height_grid = np.zeros_like(building_grid)
    
    # Assign a random height to each labeled block
    for block_label in range(1, num_blocks + 1):
        # Random height for the entire block
        block_height = random.randint(*height_range)
        
        # Assign this height to all cells in the block
        height_grid[labeled_blocks == block_label] = block_height
    
    return height_grid

# Increase building_width until building density parameter matches the chosen LCZ
for building_width in tqdm(range(1,29), desc="Matching building density"):
    road_grid = generate_full_road_network_with_width(size, lcz_type, lcz_params, road_spacing)
    
    building_grid = place_buildings_along_roads(road_grid, lcz_params, lcz_type, building_width)
    
    height_grid = assign_uniform_building_heights_with_density(building_grid, lcz_params, lcz_type)
    
    density = sum(sum(building_grid[10:-10,10:-10]))/((size-20)*(size-20))

    if (density+0.02 > lcz_params[lcz_type]["density"]) & (density < lcz_params[lcz_type]["density"]):
        print(" Final building density equals "+str(density))
        break

# Visualizing city district
plt.imshow(height_grid)

""" Diagnose sky view factor (SVF) to check if it is in line with LCZ
If it deviates too much, other parameters at the top (e.g. street width) can
be modified until building density and SVF match the chosen LCZ
""" 

from numba import njit

@njit  # JIT compile this function for speed
def calculate_svf_for_grid_point(i, j, height_grid, resolution, max_distance):
    """
    Calculate SVF for a single grid point using JIT compilation for speed.
    
    Args:
    - i, j (int): Grid coordinates of the point.
    - height_grid (np.array): 2D array of building heights.
    - resolution (int): Number of directions to sample the sky (azimuth resolution).
    - max_distance (int): Maximum distance to look for obstructions.
    
    Returns:
    - svf (float): Sky View Factor for the point (i, j).
    """
    rows, cols = height_grid.shape
    
    # Manually generate angles to avoid using unsupported 'np.linspace' with 'endpoint=False'
    angles = np.arange(0, 2 * np.pi, 2 * np.pi / resolution)
    
    max_angles = []

    for angle in angles:
        max_elevation_angle = 0
        for distance in range(1, max_distance):
            dx = int(distance * np.cos(angle))
            dy = int(distance * np.sin(angle))

            x, y = i + dx, j + dy
            if x < 0 or y < 0 or x >= rows or y >= cols:
                break

            height_diff = height_grid[x, y] - height_grid[i, j]
            dist = np.sqrt(dx**2 + dy**2)

            if dist > 0:
                elevation_angle = np.arctan(height_diff / dist)
                max_elevation_angle = max(max_elevation_angle, elevation_angle)

        sky_visibility = 1 - max(0, max_elevation_angle / (np.pi / 2))
        max_angles.append(sky_visibility)

    # Convert max_angles list to a NumPy array to calculate the mean
    return np.mean(np.array(max_angles))


def calculate_svf(height_grid, resolution=36, height_threshold=0):
    """
    Calculate Sky View Factor (SVF) for each point in the height grid, excluding building grids.
    
    Args:
    - height_grid (np.array): 2D array representing building heights.
    - resolution (int): Number of directions to sample the sky (azimuth resolution).
    - height_threshold (float): Threshold above which a grid is considered occupied by a building.
    
    Returns:
    - svf_grid (np.array): 2D array of SVF values for non-building points.
    """
    rows, cols = height_grid.shape
    svf_grid = np.zeros((rows, cols))  # Initialize the SVF grid with zeros
    max_distance = max(rows, cols)  # Maximum distance for SVF calculations

    # Create a mask for non-building grids (where height <= height_threshold)
    non_building_mask = height_grid <= height_threshold

    # Iterate over each grid point and calculate SVF for non-building cells
    for i in tqdm(range(rows), desc="Calculating SVF", unit="row"):
        for j in range(cols):
            if not non_building_mask[i, j]:
                continue  # Skip if it's a building grid
            
            svf_grid[i, j] = calculate_svf_for_grid_point(i, j, height_grid, resolution, max_distance)
    filter = svf_grid==0
    svf_grid[filter] = np.nan
    return svf_grid


def plot_svf(svf_grid):
    """
    Plot the Sky View Factor (SVF) grid.
    
    Args:
    - svf_grid (np.array): 2D array of SVF values.
    """
    plt.imshow(svf_grid, cmap='coolwarm', origin='upper')
    plt.colorbar(label="Sky View Factor")
    plt.title("Sky View Factor (SVF)")
    plt.axis('off')
    plt.show()


# Example usage with a building height grid
svf_grid = calculate_svf(height_grid, resolution=36, height_threshold=0)  # Assume buildings have height > 0
plot_svf(svf_grid)

# average SVF for the core area (excluding borders)
max_building_height = lcz_params["LCZ2"]["height_range"][1]
svf_mean = np.nanmean(svf_grid[max_building_height:-1-max_building_height,max_building_height:-1-max_building_height])


""" Export an image that e.g. can be used as input for Envi-met"""
from scipy.ndimage import label, find_objects

def export_labeled_height_grid(height_grid, filename='labeled_height_grid.png'):
    """
    Export the height grid as an image, without axes, and with a label showing the correct height 
    of each distinct building block at its most representative location.
    
    Args:
    - height_grid (np.array): A 2D array of building heights.
    - filename (str): The name of the output image file.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the height grid without axes
    ax.imshow(height_grid, cmap='viridis', origin='upper')

    # Remove axes
    ax.axis('off')

    # Label distinct building blocks
    labeled_grid, num_features = label(height_grid > 0)  # Detect distinct building blocks
    slices = find_objects(labeled_grid)  # Get the bounding slices for each labeled building block

    for idx, slice_tuple in enumerate(slices):
        # Extract the block's bounding box from the height grid
        block_slice = height_grid[slice_tuple]
        
        # Find the position (relative to the block slice) where the maximum height occurs
        block_max_pos = np.unravel_index(np.argmax(block_slice), block_slice.shape)
        
        # Translate the local position to the global position in the grid
        y_global = block_max_pos[0] + slice_tuple[0].start
        x_global = block_max_pos[1] + slice_tuple[1].start
        
        # Get the corresponding building height
        height_value = int(height_grid[y_global, x_global])
        
        # Place the label at the location of the maximum height in the building block
        ax.text(x_global, y_global, f'{height_value}', color='white', ha='center', va='center', fontsize=8)

    # Save the image
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

# Example usage: Export the height grid with non-overlapping labels showing correct building heights
export_labeled_height_grid(height_grid, filename='labeled_height_grid.png')


