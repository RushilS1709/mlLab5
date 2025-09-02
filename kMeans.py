import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
# Load dataset
df = pd.read_csv("combined_features.csv")

# Select important numeric features
features = [
    "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes",
    "avg_pkt_size", "pkt_size_entropy", "duration",
    "pkt_rate", "byte_rate"
]

X = df[features]

# Scale features for better clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans clustering with 2 clusters (VPN / Non-VPN)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

print(df[["label", "Cluster"]].head())



# Silhouette Score
sil_score = silhouette_score(X_scaled, kmeans.labels_)

# Davies–Bouldin Index
db_index = davies_bouldin_score(X_scaled, kmeans.labels_)

print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Index: {db_index:.4f}")

sil_scores = [sil_score]
db_indexes = [db_index]
inertia = []

for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    sil = silhouette_score(X_scaled, kmeans.labels_)
    db = davies_bouldin_score(X_scaled, kmeans.labels_)
    
    sil_scores.append(sil)
    db_indexes.append(db)

    inertia.append(kmeans.inertia_)
    
    print(f"k={k}: Silhouette={sil:.4f}, Davies-Bouldin={db:.4f}")



plt.figure(figsize=(8,5))
plt.plot(range(2, 20), inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()
