# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:32:39 2024

@author: colli
"""

import numpy as np

# Define grid parameters
nx = 100  # Number of points along x
ny = 100  # Number of points along y
dx = 40.0  # Grid spacing along x
dy = 40.0  # Grid spacing along y

# Create a custom terrain (e.g., a sinusoidal wave)
x = np.linspace(0, (nx-1)*dx, nx)
y = np.linspace(0, (ny-1)*dy, ny)
X, Y = np.meshgrid(x, y)

# Define the terrain: Example: A central peak using a Gaussian function
h_values = 25.0 - 15.0 * np.exp(-((X - 2000)**2 + (Y - 2000)**2) / (600**2))

#h_values = 20.0 + 0.01 * X  # Depth increases with x
#h_values = 20.0 + 2.0 * np.random.rand(ny, nx)


# Write the bathymetric data to a .dat file
with open('custom_terrain.dat', 'wb') as f:
    # Write header: nx, ny, dx, dy
    np.array([nx], dtype=np.int32).tofile(f)
    np.array([ny], dtype=np.int32).tofile(f)
    np.array([dx], dtype=np.float64).tofile(f)
    np.array([dy], dtype=np.float64).tofile(f)
    
    # Write the depth values in row-major order (flattened array)
    h_values.flatten().tofile(f)

