import numpy as np
import pandas as pd
import os

# Chemin du fichier CSV
csv_file_path = r"C:\École\BAC - 3\Q1\High Perf. Sci. Computing\local derniere Q\petite zone avec terre\Mean depth rainbow colour (no land).csv"

# Charger les données CSV depuis le fichier téléchargé, en ignorant la deuxième ligne pour les en-têtes
data = pd.read_csv(csv_file_path, skiprows=[1])

# Constantes
RADIUS_EARTH = 6371000  # Rayon moyen de la Terre en mètres
MINIMUM_ELEVATION = 0.5  # Valeur minimale d'élévation en mètres si on se trouve sur un bout de terre

# Préparer les arrays pour les coordonnées et les valeurs d'élévation
latitudes = data['latitude'].values.astype(float)
longitudes = data['longitude'].values.astype(float)
elevations = -data['elevation'].values.astype(float)

# Calculer dx et dy en mètres
dx, dy = None, None

# Trouver le premier changement de latitude
for i in range(1, len(latitudes)):
    if latitudes[i] != latitudes[i - 1]:
        dlat_deg = np.abs(latitudes[i] - latitudes[i - 1])
        dx = dlat_deg * (np.pi / 180) * RADIUS_EARTH
        break

# Trouver le premier changement de longitude
for i in range(1, len(longitudes)):
    if longitudes[i] != longitudes[i - 1]:
        dlon_deg = np.abs(longitudes[i] - longitudes[i - 1])
        avg_lat_rad = np.mean(latitudes) * (np.pi / 180)
        dy = dlon_deg * (np.pi / 180) * RADIUS_EARTH * np.cos(avg_lat_rad)
        break

nx = len(np.unique(latitudes))
ny = len(np.unique(longitudes))

# Création d'une matrice pour les élévations
elevation_matrix = np.zeros((ny, nx))
for idx, lat in enumerate(np.unique(latitudes)):
    for idy, lon in enumerate(np.unique(longitudes)):
        index = np.where((data['latitude'] == lat) & (data['longitude'] == lon))
        if index[0].size == 1:  # Assurer qu'un seul indice est trouvé
            elevation_value = elevations[index[0][0]]
            elevation_matrix[idy, idx] = elevation_value if elevation_value >= 0 else MINIMUM_ELEVATION
        elif index[0].size > 1:  # Plusieurs indices trouvés, prendre la moyenne ou une autre stratégie
            elevation_value = np.mean(elevations[index[0]])
            elevation_matrix[idy, idx] = elevation_value if elevation_value >= 0 else MINIMUM_ELEVATION

# Chemin du fichier .dat à créer dans le même répertoire que le fichier CSV
output_file_path = os.path.join(os.path.dirname(csv_file_path), 'output_data.dat')

# Écrire les données dans le fichier .dat
with open(output_file_path, 'wb') as f:
    np.array([nx], dtype=np.int32).tofile(f)
    np.array([ny], dtype=np.int32).tofile(f)
    np.array([dx], dtype=np.float64).tofile(f)
    np.array([dy], dtype=np.float64).tofile(f)
    elevation_matrix.flatten().tofile(f)