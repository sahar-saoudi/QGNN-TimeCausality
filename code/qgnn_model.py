import torch
import time
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.loader import DataLoader
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer
np.random.seed(42)


# ======================
# Donnees et pretraitement
# ======================

def load_financial_data(data_path, rels_path, num_vars):
    df = pd.read_csv(data_path, header=None)
    df.columns = [f"P{i}" for i in range(num_vars)]
    rels = pd.read_csv(rels_path, header=None)
    rels.columns = ["cause", "effect", "lag"]
    return df, rels

def create_graph_from_time_slice(t, df, rels, num_vars):
    G = nx.DiGraph()
    for i in range(num_vars):
        G.add_node(i, x=torch.tensor([df.iloc[t, i]], dtype=torch.float))
    for _, row in rels.iterrows():
        cause, effect, lag = int(row["cause"]), int(row["effect"]), int(row["lag"])
        if t - lag >= 0:
            G.add_edge(cause, effect)
    return G

def get_adjacency_target(G, num_vars):
    adj = torch.zeros((num_vars, num_vars))
    for i, j in G.edges():
        adj[i, j] = 1.0
    return adj.flatten()

def generate_dataset(df, rels, num_vars, time_range):
    data_list = []
    for t in time_range:
        G = create_graph_from_time_slice(t, df, rels, num_vars)
        if G.number_of_edges() == 0:
            continue
        pyg_data = from_networkx(G)
        pyg_data.x = torch.stack([G.nodes[i]['x'] for i in range(len(G.nodes))])
        pyg_data.y = get_adjacency_target(G, num_vars)
        data_list.append(pyg_data)
    return data_list


# ======================
# Couche Quantique (Qiskit)
# ======================

class QiskitQuantumLayer(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        self.weights = nn.Parameter(torch.randn(n_qubits))
        self.backend = AerSimulator()

    def circuit(self, x, weights):
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.ry(float(x[i]), i)
        for i in range(self.n_qubits - 1):
            qc.cx(i, i+1)
        for i in range(self.n_qubits):
            qc.rz(float(weights[i]), i)
        qc.save_statevector()
        return qc

    def forward(self, x):
        x = x.view(-1)
        qc = self.circuit(x, self.weights)

        compiled = transpile(qc, self.backend)
        result = self.backend.run(compiled).result()
        statevector = result.get_statevector()
        probs = np.abs(statevector) ** 2
        reduced = torch.tensor(probs[:self.n_qubits], dtype=torch.float)
        return reduced
    


# ======================
# Modèle QGNN complet
# ======================

class QGNNModel(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.quantum = QiskitQuantumLayer(num_nodes)
        self.fc = nn.Linear(num_nodes, num_nodes * num_nodes)

    def forward(self, data):
        q_out = self.quantum(data.x)
        logits = self.fc(q_out)
        return logits



# ======================
# Entraînement
# ======================

def train(model, loader, epochs=50, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        total_loss = 0
        for data in loader:
            out = model(data)
            loss = loss_fn(out, data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss / len(loader):.4f}")



# ======================
# Visualization graphes
# ======================

def visualize_graphs(true_matrix, pred_matrix):
    G_true = nx.from_numpy_array(true_matrix.numpy(), create_using=nx.DiGraph)
    G_pred = nx.from_numpy_array(pred_matrix.numpy(), create_using=nx.DiGraph)
    pos = nx.spring_layout(G_true, seed=42)

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    nx.draw(G_true, pos, with_labels=True, node_color='lightblue', edge_color='blue')
    plt.title("Graphe causal réel")

    plt.subplot(1, 2, 2)
    nx.draw(G_pred, pos, with_labels=True, node_color='lightyellow', edge_color='orange')
    plt.title("Graphe causal prédit")
    plt.tight_layout()
    plt.show()



# ======================
# Évaluation
# ======================

def evaluate(model, loader, num_nodes, threshold=0.5):
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for data in loader:
            logits = model(data)
            preds = torch.sigmoid(logits) > threshold
            all_preds.extend(preds.int().tolist())
            all_trues.extend(data.y.int().tolist())

    precision = precision_score(all_trues, all_preds, zero_division=0)
    recall = recall_score(all_trues, all_preds, zero_division=0)
    f1 = f1_score(all_trues, all_preds, zero_division=0)
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
    
    inf_matrix = preds.int().view(num_nodes, num_nodes)
    true_matrix = data.y.int().view(num_nodes, num_nodes)


    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(true_matrix, annot=True, cmap="Blues")
    plt.title("Matrice causale réelle")
    plt.subplot(1, 2, 2)
    sns.heatmap(inf_matrix, annot=True, cmap="Oranges")
    plt.title("Matrice causale prédite")
    plt.tight_layout()
    plt.show()

    # Visualisation des graphes causaux
    visualize_graphs(true_matrix, inf_matrix)



# ======================
# Visualisation circuit
# ======================
def display_final_example_circuit(n_qubits, trained_weights=None, input_data=None):
    from qiskit import QuantumCircuit
    import matplotlib.pyplot as plt

    qc = QuantumCircuit(n_qubits)

    # Données d'exemple
    if input_data is None:
        input_data = np.random.uniform(0, np.pi, size=n_qubits)

    if trained_weights is None:
        trained_weights = np.random.uniform(0, 2*np.pi, size=n_qubits)

    # Embedding des features
    for i in range(n_qubits):
        qc.ry(float(input_data[i]), i)

    # Entanglement simple
    for i in range(n_qubits - 1):
        qc.cx(i, i+1)

    # Paramètres appris
    for i in range(n_qubits):
        qc.rz(float(trained_weights[i]), i)

    # Affichage du circuit
    print(qc.draw(output='text'))

    fig = qc.draw(output='mpl', scale=0.5, fold=20)
    plt.title("Circuit quantique illustratif utilisé dans le QGNN")
    fig.savefig("circuit_qgnn.png", dpi=300, bbox_inches='tight')
    plt.show()



# ======================
# Pipeline principal
# ======================

if __name__ == "__main__":
    start_time = time.time()
    data_path = "/Users/mohamedsaoudi/Desktop/stage/Implementation/dataset1/FinanceCPT/returns/random-rels_20_1A_returns30007000.csv"
    rels_path = "/Users/mohamedsaoudi/Desktop/stage/Implementation/dataset1/FinanceCPT/relationships/random-rels_20_1A.csv"
    num_vars = 25 
    df, rels = load_financial_data(data_path, rels_path, num_vars)
    start_t = max(rels["lag"])
    dataset = generate_dataset(df, rels, num_vars, range(start_t + 1, start_t + 8))
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = QGNNModel(num_nodes=num_vars)
    train(model, loader, epochs=40)
    evaluate(model, loader, num_nodes=num_vars)

    print("Affichage du circuit quantique final illustratif...")
    example_data = next(iter(loader))
    x_input = example_data.x.view(-1).detach().numpy()
    trained_weights = model.quantum.weights.detach().numpy()

    display_final_example_circuit(num_vars, trained_weights, x_input)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f" Total runtime: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
