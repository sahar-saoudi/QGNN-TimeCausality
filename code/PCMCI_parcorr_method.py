#== PCMCI parcorr==

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI
from sklearn.metrics import precision_score, recall_score, f1_score

# === Chemins des fichiers ===
data_path = "/Users/mohamedsaoudi/Desktop/stage/Implementation/dataset1/FinanceCPT/returns/random-rels_20_1A_returns30007000.csv"
rels_path = "/Users/mohamedsaoudi/Desktop/stage/Implementation/dataset1/FinanceCPT/relationships/random-rels_20_1A.csv"

# === Chargement des données ===
df = pd.read_csv(data_path, header=None)
data = df.to_numpy()

# === Formatage pour Tigramite ===
# shape: (timepoints, variables)
dataframe = pp.DataFrame(data)

# === Matrice vérité (ground truth) ===
num_vars = data.shape[1]
rels = pd.read_csv(rels_path, header=None)
rels.columns = ["cause", "effect", "lag"]
target = np.zeros((num_vars, num_vars), dtype=int)
for _, row in rels.iterrows():
    target[int(row["cause"]), int(row["effect"])] = 1

# === Initialisation PCMCI ===
parcorr = ParCorr(significance='analytic')
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr)

# === Paramètres ===
max_lag = 3
alpha_level = 0.05  # seuil de significativité

# === Exécution de PCMCI ===
results = pcmci.run_pcmci(tau_max=max_lag, pc_alpha=None)

# === Seuil sur les p-values corrigées ===
q_matrix = results['p_matrix']
val_matrix = results['val_matrix']
adj_pred = np.zeros((num_vars, num_vars), dtype=int)

for i in range(num_vars):
    for j in range(num_vars):
        for lag in range(1, max_lag + 1):
            pval = q_matrix[i, j, lag]
            if pval < alpha_level:
                adj_pred[j, i] = 1  # j at t-lag → i at t

# === Évaluation ===
mask = ~np.eye(num_vars, dtype=bool)
y_true = target[mask]
y_pred = adj_pred[mask]

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\n=== Résultats avec PCMCI ===")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

# === Visualisation heatmaps ===
import seaborn as sns

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(target, ax=axs[0], cmap="Blues", square=True, cbar=True)
axs[0].set_title("Matrice Causale Vérité")

sns.heatmap(adj_pred, ax=axs[1], cmap="Blues", square=True, cbar=True)
axs[1].set_title("Matrice Prédite (PCMCI)")

plt.tight_layout()
plt.show()
