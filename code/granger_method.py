# === Granger ===

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# === PARAMÈTRES ===
data_path = "/Users/mohamedsaoudi/Desktop/stage/Implementation/dataset1/FinanceCPT/returns/random-rels_20_1A_returns30007000.csv"
rels_path = "/Users/mohamedsaoudi/Desktop/stage/Implementation/dataset1/FinanceCPT/relationships/random-rels_20_1A.csv"
max_lag = 3
alpha = 0.05  # seuil global pour FDR

# === Chargement des données ===
df = pd.read_csv(data_path, header=None)
df.columns = [f"P{i}" for i in range(df.shape[1])]
num_vars = df.shape[1]

rels = pd.read_csv(rels_path, header=None)
rels.columns = ["cause", "effect", "lag"]

# === Matrice cible (vérité terrain) ===
target = np.zeros((num_vars, num_vars), dtype=int)
for _, row in rels.iterrows():
    target[int(row["cause"]), int(row["effect"])] = 1

# === Collecte des p-values pour chaque paire (i, j) ===
all_p_values = []
pairs = []

print("Testing Granger causality with FDR correction...")
for i in range(num_vars):
    for j in range(num_vars):
        if i == j:
            continue
        try:
            data_pair = np.column_stack([df.iloc[:, j].values, df.iloc[:, i].values])  # j causes i
            result = grangercausalitytests(data_pair, maxlag=max_lag, verbose=False)
            min_p = min(result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1))
            all_p_values.append(min_p)
        except:
            all_p_values.append(1.0)
        pairs.append((i, j))

# === Correction FDR (Benjamini-Hochberg) ===
rejected, pvals_corrected, _, _ = multipletests(all_p_values, alpha=alpha, method='fdr_bh')

# === Matrice prédite corrigée ===
predicted = np.zeros((num_vars, num_vars), dtype=int)
for idx, (i, j) in enumerate(pairs):
    if rejected[idx]:
        predicted[i, j] = 1

# === Évaluation ===
mask = ~np.eye(num_vars, dtype=bool)
y_true = target[mask]
y_pred = predicted[mask]

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\n=== Résultats avec FDR (Granger) ===")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

# === Visualisation ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(target, ax=axes[0], cmap="Blues", square=True, cbar=True)
axes[0].set_title("Matrice Causale Vérité")

sns.heatmap(predicted, ax=axes[1], cmap="Blues", square=True, cbar=True)
axes[1].set_title("Matrice Prédite (Granger + FDR)")

plt.tight_layout()
plt.show()

# === Visualisation des graphes orientés ===
def visualize_causal_graphs(target_matrix, predicted_matrix, node_labels=None):
    if node_labels is None:
        node_labels = [f"P{i}" for i in range(target_matrix.shape[0])]

    # Création des graphes
    G_true = nx.DiGraph()
    G_pred = nx.DiGraph()

    for i in range(len(node_labels)):
        G_true.add_node(i, label=node_labels[i])
        G_pred.add_node(i, label=node_labels[i])

    for i in range(target_matrix.shape[0]):
        for j in range(target_matrix.shape[1]):
            if target_matrix[i, j] == 1:
                G_true.add_edge(i, j)
            if predicted_matrix[i, j] == 1:
                G_pred.add_edge(i, j)

    # Disposition fixe pour que les graphes soient comparables
    pos = nx.spring_layout(G_true, seed=42)

    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    # Graphe de vérité
    nx.draw_networkx_nodes(G_true, pos, ax=axs[0], node_color='skyblue', node_size=800)
    nx.draw_networkx_edges(G_true, pos, ax=axs[0], edge_color='black', arrows=True, width=2)
    nx.draw_networkx_labels(G_true, pos, ax=axs[0], labels={i: node_labels[i] for i in range(len(node_labels))})
    axs[0].set_title("Graphe Causal – Vérité Terrain")
    axs[0].axis('off')

    # Graphe prédit
    nx.draw_networkx_nodes(G_pred, pos, ax=axs[1], node_color='lightgreen', node_size=800)
    nx.draw_networkx_edges(G_pred, pos, ax=axs[1], edge_color='gray', arrows=True, width=2)
    nx.draw_networkx_labels(G_pred, pos, ax=axs[1], labels={i: node_labels[i] for i in range(len(node_labels))})
    axs[1].set_title("Graphe Causal – Méthode Granger + FDR")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

# === Appel de la fonction avec les matrices ===
visualize_causal_graphs(target, predicted)
