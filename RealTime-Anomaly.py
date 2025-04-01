import os
import time
import torch
import pandas as pd
import networkx as nx
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Start overall timer
overall_start_time = time.time()

# -------------------------------
# 1. Data Loading and Preprocessing
# -------------------------------
file_path = r"D:\CV Neda\New folder\dataset\r4.2 (2)\r4.2\New folder\merged_data.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

start_time = time.time()
df = pd.read_csv(file_path, low_memory=False, nrows=10000)

# Convert date to datetime and extract time features
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df["hour"] = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.dayofweek

# Encode categorical variables
activity_encoder = LabelEncoder()
role_encoder = LabelEncoder()
df["activity_encoded"] = activity_encoder.fit_transform(df["activity"])
df["role_encoded"] = role_encoder.fit_transform(df["role"])

data_loading_time = time.time() - start_time

# -------------------------------
# 2. Convert User-PC to Integer IDs
# -------------------------------
start_time = time.time()
unique_nodes = set(df["user"].tolist() + df["pc"].tolist())
global_node_mapping = {node: i for i, node in enumerate(unique_nodes)}
df["user_id"] = df["user"].map(global_node_mapping)
df["pc_id"] = df["pc"].map(global_node_mapping)

node_mapping_time = time.time() - start_time

# -------------------------------
# 3. Define GNN Model
# -------------------------------
class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = pyg_nn.SAGEConv(input_dim, hidden_dim)
        self.conv2 = pyg_nn.SAGEConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x

# -------------------------------
# 4. Setup Model and Training Parameters
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraphSAGE(input_dim=4, hidden_dim=64, output_dim=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()
num_epochs = 50  # Used for initial training and re-challenging

# -------------------------------
# 5. Anomaly Detection Setup
# -------------------------------
iso_forest = IsolationForest(contamination=0.06, random_state=42)
scaler = StandardScaler()
node_features_all = df[["hour", "day_of_week", "activity_encoded", "role_encoded"]].values
scaler.fit(node_features_all)

# -------------------------------
# 6. Process Logs by Time Slots (Dynamic Sliding Window)
# -------------------------------
def filter_by_date_range(df, start_date, minutes):
    df['date'] = pd.to_datetime(df['date'])
    end_date = start_date + pd.Timedelta(minutes=minutes)
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    return filtered_df

def init_time_slot(df, start_date, minutes):
    res = filter_by_date_range(df, start_date, minutes)
    return [start_date, res]

def next_time_slot(df, start_date, minutes, slide):
    new_start_date = start_date + pd.Timedelta(minutes=slide)
    result_df = filter_by_date_range(df, new_start_date, minutes)
    return [new_start_date, result_df]

def data_stream(df, start_date, initial_minutes=10, initial_slide=5):
    minutes = initial_minutes
    slide = initial_slide
    res = init_time_slot(df, start_date, minutes)
    total_slots = ((df['date'].max() - start_date) // pd.Timedelta(minutes=slide)) + 1
    with tqdm(total=total_slots, desc="Processing time slots") as pbar:
        yield res[1], res[0], minutes, slide
        while not res[1].empty:
            # Dynamic window logic
            if res[0] >= pd.Timestamp("2021-01-03"):  # Example threshold date
                minutes = 60  # Switch to 1-hour windows
                slide = 15    # Slide every 15 minutes
            res = next_time_slot(df, res[0], minutes, slide)
            yield res[1], res[0], minutes, slide
            time.sleep(1)  # Simulate real-time delay
            pbar.update(1)

# -------------------------------
# 7. Training Function
# -------------------------------
def train_model(model, graph_data, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        embeddings = model(graph_data.x, graph_data.edge_index)
        loss = criterion(embeddings, graph_data.x)
        loss.backward()
        optimizer.step()
    return model

# Initialize variables
start_date = df["date"].min()
all_node_anomalies = []
window_metrics = []  # For granularity comparison
training_time_total = 0
graph_creation_time_total = 0
first_slot = True
slot_count = 0

# Process logs by time slots
for time_slot_df, current_slot_time, minutes, slide in data_stream(df, start_date):
    if time_slot_df.empty:
        continue

    slot_count += 1
    start_time = time.time()

    # Create a graph for the current time slot
    G = nx.DiGraph()
    unique_nodes = set(time_slot_df["user_id"].tolist() + time_slot_df["pc_id"].tolist())
    node_index_map = {node: idx for idx, node in enumerate(unique_nodes)}

    # Build edge list and graph
    edge_list = []
    for _, row in time_slot_df.iterrows():
        user_idx = node_index_map[row["user_id"]]
        pc_idx = node_index_map[row["pc_id"]]
        edge_list.append([user_idx, pc_idx])
        G.add_edge(row["user_id"], row["pc_id"], activity=row["activity_encoded"])

    graph_creation_time = time.time() - start_time
    graph_creation_time_total += graph_creation_time

    # Create edge_index tensor
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)

    # Prepare node features for all unique nodes
    node_features_list = []
    for node in unique_nodes:
        node_data = time_slot_df[(time_slot_df["user_id"] == node) | (time_slot_df["pc_id"] == node)]
        if not node_data.empty:
            features = node_data[["hour", "day_of_week", "activity_encoded", "role_encoded"]].iloc[-1].values
        else:
            features = [0, 0, 0, 0]
        node_features_list.append(features)

    node_features = scaler.transform(node_features_list)
    node_features = torch.tensor(node_features, dtype=torch.float).to(device)

    # Create PyG Graph Data Object
    graph_data = Data(x=node_features, edge_index=edge_index).to(device)
    graph_data.num_nodes = len(unique_nodes)

    # Train or infer
    if first_slot or (slot_count % 12 == 0):  # Re-train every 12 slots
        if slot_count % 12 == 0:
            print(f"Re-evaluating batch training at slot {slot_count}...")
        start_time = time.time()
        model = train_model(model, graph_data, num_epochs)
        training_time = time.time() - start_time
        training_time_total += training_time
        first_slot = False
        try:
            model.eval()
            with torch.no_grad():
                embeddings = model(graph_data.x, graph_data.edge_index).cpu().numpy()
        except Exception as e:
            print(f"Embedding inference failed: {e}")
            continue
    else:
        try:
            model.eval()
            with torch.no_grad():
                embeddings = model(graph_data.x, graph_data.edge_index).cpu().numpy()
        except Exception as e:
            print(f"Embedding inference failed: {e}")
            continue

    # Calculate anomaly scores and track nodes
    iso_forest.fit(embeddings)
    iso_anomaly_scores = -iso_forest.decision_function(embeddings)

    # Store anomaly scores with node info
    for i, node_id in enumerate(unique_nodes):
        score = iso_anomaly_scores[i]
        all_node_anomalies.append({
            "timestamp": current_slot_time,
            "node_id": node_id,
            "score": score,
        })

    # Real-time alerting and metrics
    threshold = pd.Series([entry["score"] for entry in all_node_anomalies]).quantile(0.84)
    num_anomalies = sum(score > threshold for score in iso_anomaly_scores)
    for i, score in enumerate(iso_anomaly_scores):
        if score > threshold:
            print(f"ALERT: Anomalous behavior at {current_slot_time} for node {list(unique_nodes)[i]} with score {score}")

    # Log window metrics
    window_metrics.append({
        "timestamp": current_slot_time,
        "window_size": minutes,
        "num_events": len(time_slot_df),
        "num_anomalies": num_anomalies,
    })

# -------------------------------
# 8. Save and Visualize Anomalies
# -------------------------------
anomaly_df = pd.DataFrame(all_node_anomalies)
anomaly_df["anomaly_label"] = (anomaly_df["score"] > threshold).astype(int)
anomaly_df.to_csv("gnn_anomalies_incremental.csv", index=False)

metrics_df = pd.DataFrame(window_metrics)
metrics_df.to_csv("window_metrics.csv", index=False)

# Visualization Dashboard
plt.figure(figsize=(15, 10))

# Anomaly Score Distribution by Window Size
plt.subplot(2, 2, 1)
for window_size in metrics_df["window_size"].unique():
    scores = anomaly_df[anomaly_df["timestamp"].isin(
        metrics_df[metrics_df["window_size"] == window_size]["timestamp"]
    )]["score"]
    sns.kdeplot(scores, label=f"{window_size} min", alpha=0.5)
plt.axvline(threshold, color="black", linestyle="--", label="Threshold")
plt.title("Anomaly Score Distribution by Window Size")
plt.xlabel("Anomaly Score")
plt.ylabel("Density")
plt.legend()

# Detection Latency vs. Granularity
plt.subplot(2, 2, 2)
sns.scatterplot(data=metrics_df, x="window_size", y="num_events", size="num_anomalies", hue="num_anomalies")
plt.title("Events and Anomalies vs. Window Size")
plt.xlabel("Window Size (minutes)")
plt.ylabel("Number of Events")

# Number of Anomalous Nodes per Hour
metrics_df["hour"] = metrics_df["timestamp"].dt.floor("H")
hourly_anomalies = metrics_df.groupby("hour")["num_anomalies"].sum().reset_index()
plt.subplot(2, 2, 3)
sns.lineplot(data=hourly_anomalies, x="hour", y="num_anomalies")
plt.title("Anomalous Events per Hour")
plt.xlabel("Hour")
plt.ylabel("Number of Anomalies")
plt.xticks(rotation=45)

# Anomaly Rate Over Time
plt.subplot(2, 2, 4)
metrics_df["anomaly_rate"] = metrics_df["num_anomalies"] / metrics_df["num_events"]
sns.lineplot(data=metrics_df, x="timestamp", y="anomaly_rate", hue="window_size")
plt.title("Anomaly Rate Over Time by Window Size")
plt.xlabel("Timestamp")
plt.ylabel("Anomaly Rate")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("\nAnomaly Detection Complete. Results saved to 'gnn_anomalies_incremental.csv' and 'window_metrics.csv'.")

# Calculate overall time
overall_time = time.time() - overall_start_time

# -------------------------------
# 9. Export Timing Results
# -------------------------------
log_file_path = os.path.join(os.path.dirname(file_path), os.path.basename(file_path).replace(".csv", "_log.txt"))
with open(log_file_path, "w") as log_file:
    log_file.write("Timing Results:\n")
    log_file.write(f"Overall Time: {overall_time:.2f} seconds\n")
    log_file.write(f"Data Loading Time: {data_loading_time:.2f} seconds\n")
    log_file.write(f"Node Mapping Time: {node_mapping_time:.2f} seconds\n")
    log_file.write(f"Graph Creation Time (Total): {graph_creation_time_total:.2f} seconds\n")
    log_file.write(f"Training Time (Total): {training_time_total:.2f} seconds\n")

print(f"Timing results saved to {log_file_path}") HOW CAN I PUT IN MY GITHUB