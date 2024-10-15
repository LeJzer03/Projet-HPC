# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 19:03:29 2024

@author: colli
"""

import matplotlib.pyplot as plt


delta_t_values = [2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]  # valeurs de delta t
execution_times = [1.20848, 2.38453, 4.6754, 21.0086, 38.9852, 86.5015, 215.305, 371.707, 739.193, 1756.7, 3489.67, 7028.01]  # temps d'exécution en secondes pour chaque delta t

# Créer le graphique
plt.figure(figsize=(8, 6))
plt.plot(delta_t_values, execution_times, marker='o', linestyle='-', color='b')

# Ajouter les titres et étiquettes
plt.title('Temps d\'exécution en fonction de Delta t', fontsize=16)
plt.xlabel('Delta t (s)', fontsize=14)
plt.ylabel('Temps d\'exécution (s)', fontsize=14)

# Ajouter une grille
plt.grid(True)

# Afficher le graphique
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(delta_t_values, execution_times, marker='o', linestyle='-', color='b')

# Utiliser une échelle logarithmique pour les deux axes
plt.xscale('log')
plt.yscale('log')

# Ajouter les titres et étiquettes
plt.title('Temps d\'exécution en fonction de Delta t (échelle logarithmique)', fontsize=16)
plt.xlabel('Delta t (s)', fontsize=14)
plt.ylabel('Temps d\'exécution (s)', fontsize=14)

# Ajouter une grille
plt.grid(True, which="both", ls="--")

# Afficher le graphique
plt.show()