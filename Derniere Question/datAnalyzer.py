import numpy as np
import os

def read_dat_file(file_path):
    """
    Lit un fichier .dat contenant des données d'élévation géographiques
    et les convertit en une matrice numpy.

    Args:
    file_path (str): Chemin complet vers le fichier .dat.

    Returns:
    tuple: Contient la matrice des élévations, et les valeurs de dx et dy.
    """
    with open(file_path, 'rb') as file:
        # Lire l'en-tête contenant nx, ny, dx, dy
        nx = np.fromfile(file, dtype=np.int32, count=1)[0]
        ny = np.fromfile(file, dtype=np.int32, count=1)[0]
        dx = np.fromfile(file, dtype=np.float64, count=1)[0]
        dy = np.fromfile(file, dtype=np.float64, count=1)[0]

        # Lire les données d'élévation
        elevations = np.fromfile(file, dtype=np.float64, count=nx*ny)
        elevation_matrix = elevations.reshape(ny, nx)  # Remettre en forme la matrice selon ny et nx

    return elevation_matrix, dx, dy

# Emplacement du fichier .dat (ajustez selon votre configuration)
dat_file_path = r"C:\École\BAC - 3\Q1\High Perf. Sci. Computing\local derniere Q\petite zone avec terre\output_data.dat"

# Appel de la fonction pour lire le fichier .dat
elevation_matrix, dx, dy = read_dat_file(dat_file_path)

# Afficher les résultats pour vérification
print("dx:", dx, "meters")
print("dy:", dy, "meters")
print("Elevation matrix shape:", elevation_matrix.shape)
print("Elevation matrix sample data:\n", elevation_matrix)
