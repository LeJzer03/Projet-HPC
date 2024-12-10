import numpy as np
import pandas as pd

# Charger les données CSV
data = pd.read_csv('/path/to/your/file.csv')  # Mettez ici le chemin de votre fichier CSV

# Constantes
RADIUS_EARTH = 6371000  # Rayon moyen de la Terre en mètres

# Préparer les arrays pour les coordonnées et les valeurs d'élévation
latitudes = data['latitude'].values
longitudes = data['longitude'].values
elevations = data['elevation'].values

# Calculer dx et dy en mètres
# La différence de latitude en degrés à mètres
if len(latitudes) > 1:
    dlat_deg = np.abs(latitudes[1] - latitudes[0])  # en degrés
    dx = dlat_deg * (np.pi / 180) * RADIUS_EARTH  # Conversion en radians et multiplication par le rayon de la Terre

# La différence de longitude en degrés à mètres, ajustée pour la latitude moyenne
if len(longitudes) > 1:
    dlon_deg = np.abs(longitudes[1] - longitudes[0])
    avg_lat_rad = np.mean(latitudes) * (np.pi / 180)  # Conversion de la latitude moyenne en radians
    dy = dlon_deg * (np.pi / 180) * RADIUS_EARTH * np.cos(avg_lat_rad)

nx = len(np.unique(latitudes))  # Nombre de points uniques en latitude
ny = len(np.unique(longitudes))  # Nombre de points uniques en longitude

# Création d'une matrice pour les élévations
elevation_matrix = elevations.reshape(ny, nx)

# Chemin du fichier .dat à créer
output_file_path = 'output_data.dat'

# Écrire les données dans le fichier .dat
with open(output_file_path, 'wb') as f:
    # Écrire l'en-tête : nx, ny, dx, dy
    np.array([nx], dtype=np.int32).tofile(f)
    np.array([ny], dtype=np.int32).tofile(f)
    np.array([dx], dtype=np.float64).tofile(f)
    np.array([dy], dtype=np.float64).tofile(f)

    # Écrire les valeurs d'élévation en ordre de rangée (tableau aplati)
    elevation_matrix.flatten().tofile(f)
