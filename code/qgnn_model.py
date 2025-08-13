"""
Quantum Graph Neural Network (QGNN) for Causal Structure Learning
==================================================================

This code implements a hybrid quantum-classical model to infer causal graphs
from time series data. The pipeline includes:

1. Loading a time series dataset and corresponding ground truth causal structure.
2. Converting each time slice into a PyTorch Geometric graph representation.
3. Passing the node features through a parameterized quantum circuit (Qiskit).
4. Using a small feed-forward layer to predict the adjacency matrix (causal links).
5. Training and evaluating the model with visualization of results.

Author: Sahar Saoudi
"""

# =======================
# 1. Import Dependencies
# =======================
import torch
import time
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

np.random.seed(42)  # Reproducibility


# =====================================================
# 2. Data Loading and Graph Construction
# =====================================================
def load_data(data_path, rels_path, num_vars):
    """
    Load time series and causal relationship data.

    Args:
        data_path (str): Path to the CSV containing time series (variables in columns).
        rels_path (str): Path to CSV containing ground truth causal edges.
        num_vars (int): Number of variables (nodes) in the graph.

    Returns:
        df (DataFrame): Time series data.
        rels (DataFrame): Causal relationships (cause, effect, lag).
    """
    df = pd.read_csv(data_path, header=None)
    df.columns = [f"P{i}" for i in range(num_vars)]
    rels = pd.read_csv(rels_path, header=None)
    rels.columns = ["cause", "effect", "lag"]
    return df, rels


def create_graph_from_time_slice(t, df, rels, num_vars):
    """
    Create a directed graph for a specific time step using ground truth lags.

    Args:
        t (int): Time index.
        df (DataFrame): Time series data.
        rels (DataFrame): Ground truth causal relationships.
        num_vars (int): Number of variables/nodes.

    Returns:
        G (nx.DiGraph): Directed graph with node features and edges.
    """
    G = nx.DiGraph()
    # Add nodes with their current value as a feature
    for i in range(num_vars):
        G.add_node(i, x=torch.tensor([df.iloc[t, i]], dtype=torch.float))

    # Add edges if lag condition is satisfied
    for _, row in rels.iterrows():
        cause, effect, lag = int(row["cause"]), int(row["effect"]), int(row["lag"])
        if t - lag >= 0:
            G.add_edge(cause, effect)

    return G


def get_adjacency_target(G, num_vars):
    """
    Convert a NetworkX graph into a flattened adjacency matrix tensor.

    Args:
        G (nx.DiGraph): Input graph.
        num_vars (int): Number of variables/nodes.

    Returns:
        Tensor: Flattened adjacency matrix (size = num_vars * num_vars).
    """
    adj = torch.zeros((num_vars, num_vars))
    for i, j in G.edges():
        adj[i, j] = 1.0
    return adj.flatten()


def generate_dataset(df, rels, num_vars, time_range):
    """
    Generate a dataset of PyTorch Geometric Data objects from time slices.

    Args:
        df (DataFrame): Time series data.
        rels (DataFrame): Ground truth causal edges.
        num_vars (int): Number of variables.
        time_range (iterable): Time steps to convert into graphs.

    Returns:
        list[Data]: List of PyTorch Geometric graph objects.
    """
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


# =====================================================
# 3. Quantum Layer Definition
# =====================================================
class QiskitQuantumLayer(nn.Module):
    """
    A parameterized quantum circuit layer using Qiskit.

    - Applies RY rotations to encode input features.
    - Applies a chain of CNOT gates for entanglement.
    - Applies RZ rotations with trainable weights.
    - Output : probability amplitudes as feature vectors.
    """
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        self.weights = nn.Parameter(torch.randn(n_qubits))
        self.backend = AerSimulator()

    def circuit(self, x, weights):
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.ry(float(x[i]), i)  # Encode data
        for i in range(self.n_qubits - 1):
            qc.cx(i, i+1)         # Entanglement
        for i in range(self.n_qubits):
            qc.rz(float(weights[i]), i)  # Parameterized rotation
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


# =====================================================
# 4. QGNN Model
# =====================================================
class QGNNModel(nn.Module):
    """
    Quantum Graph Neural Network model.

    Structure:
        Quantum Layer -> Linear Layer -> Adjacency Prediction
    """
    def __init__(self, num_nodes):
        super().__init__()
        self.quantum = QiskitQuantumLayer(num_nodes)
        self.fc = nn.Linear(num_nodes, num_nodes * num_nodes)

    def forward(self, data):
        q_out = self.quantum(data.x)
        logits = self.fc(q_out)
        return logits


# =====================================================
# 5. Training Function
# =====================================================
def train(model, train_loader, val_loader, epochs=50, lr=0.01):
    """
    Train the QGNN model.

    Args:
        model (nn.Module): QGNN model instance.
        train_loader (DataLoader): Training dataset loader.
        val_loader (DataLoader): Validation dataset loader.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.

    Returns:
        (list, list): Training and validation loss histories.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for data in train_loader:
            out = model(data)
            loss = loss_fn(out, data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                out = model(data)
                loss = loss_fn(out, data.y)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

    return train_losses, val_losses


# =====================================================
# 6. Plotting & Visualization
# =====================================================
def plot_losses(train_losses, val_losses):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Loss Curves")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_graphs(true_matrix, pred_matrix):
    """Draw true vs. predicted causal graphs."""
    G_true = nx.from_numpy_array(true_matrix.numpy(), create_using=nx.DiGraph)
    G_pred = nx.from_numpy_array(pred_matrix.numpy(), create_using=nx.DiGraph)
    pos = nx.spring_layout(G_true, seed=42)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    nx.draw(G_true, pos, with_labels=True, node_color='lightblue', edge_color='blue')
    plt.title("True Causal Graph")
    plt.subplot(1, 2, 2)
    nx.draw(G_pred, pos, with_labels=True, node_color='lightyellow', edge_color='orange')
    plt.title("Predicted Causal Graph")
    plt.tight_layout()
    plt.show()


# =====================================================
# 7. Evaluation
# =====================================================
def evaluate(model, loader, num_nodes, threshold=0.5):
    """
    Evaluate model performance and visualize results.

    Metrics: Precision, Recall, F1, Accuracy + Confusion Matrix
    """
    model.eval()
    all_preds, all_trues = [], []

    with torch.no_grad():
        for data in loader:
            logits = model(data)
            preds = torch.sigmoid(logits) > threshold
            all_preds.extend(preds.int().tolist())
            all_trues.extend(data.y.int().tolist())

    # Classification metrics
    precision = precision_score(all_trues, all_preds, zero_division=0)
    recall = recall_score(all_trues, all_preds, zero_division=0)
    f1 = f1_score(all_trues, all_preds, zero_division=0)
    accuracy = accuracy_score(all_trues, all_preds)

    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, Accuracy: {accuracy:.2f}")

    # Confusion matrix
    cm = confusion_matrix(all_trues, all_preds)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    # Heatmaps
    data_example = next(iter(loader))
    with torch.no_grad():
        logits = model(data_example)
        preds = (torch.sigmoid(logits) > threshold).int()
    inf_matrix = preds.view(num_nodes, num_nodes)
    true_matrix = data_example.y.view(num_nodes, num_nodes)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(true_matrix, annot=True, cmap="Blues")
    plt.title("True Causal Matrix")
    plt.subplot(1, 2, 2)
    sns.heatmap(inf_matrix, annot=True, cmap="Oranges")
    plt.title("Predicted Causal Matrix")
    plt.tight_layout()
    plt.show()

    # Graph visualization
    visualize_graphs(true_matrix, inf_matrix)


# =====================================================
# 8. Quantum Circuit Display
# =====================================================
def display_final_example_circuit(n_qubits, trained_weights=None, input_data=None):
    """
    Display the parameterized quantum circuit used in the QGNN.
    """
    qc = QuantumCircuit(n_qubits)
    if input_data is None:
        input_data = np.random.uniform(0, np.pi, size=n_qubits)
    if trained_weights is None:
        trained_weights = np.random.uniform(0, 2*np.pi, size=n_qubits)
    for i in range(n_qubits):
        qc.ry(float(input_data[i]), i)
    for i in range(n_qubits - 1):
        qc.cx(i, i+1)
    for i in range(n_qubits):
        qc.rz(float(trained_weights[i]), i)
    fig = qc.draw(output='mpl', scale=0.5, fold=20)
    plt.title("QGNN Quantum Circuit")
    fig.savefig("circuit_qgnn.png", dpi=300, bbox_inches='tight')
    plt.show()


# =====================================================
# 9. Main Script
# =====================================================
if __name__ == "__main__":
    start_time = time.time()

    # Paths to dataset and ground truth
    data_path = "/path/to/timeseries.csv"
    rels_path = "/path/to/ground_truth.csv"
    num_vars = 5 # it depends on the number of variables in your file

    # Load and prepare dataset
    df, rels = load_data(data_path, rels_path, num_vars)
    start_t = max(rels["lag"])
    dataset = generate_dataset(df, rels, num_vars, range(start_t + 1, start_t + 8))

    # Train/validation split
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    # Initialize and train model
    model = QGNNModel(num_nodes=num_vars)
    train_losses, val_losses = train(model, train_loader, val_loader, epochs=40)

    # Plot losses and evaluate
    plot_losses(train_losses, val_losses)
    evaluate(model, val_loader, num_nodes=num_vars)

    # Display final quantum circuit
    example_data = next(iter(val_loader))
    x_input = example_data.x.view(-1).detach().numpy()
    trained_weights = model.quantum.weights.detach().numpy()
    display_final_example_circuit(num_vars, trained_weights, x_input)

    # Execution time
    elapsed = time.time() - start_time
    print(f"Total runtime: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
