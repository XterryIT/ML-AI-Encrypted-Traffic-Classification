from sklearnex import patch_sklearn
patch_sklearn()

import numpy as np
import pandas as pd
from datetime import datetime
import time
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Metrics (Unsupervised)
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Clustering Algorithms
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.ensemble import IsolationForest # Tree-based (Anomaly Detection)

# Dimensionality Reduction / Manifold Learning
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import LocallyLinearEmbedding

# --- SETTINGS ---
BASE_DIR = "models/unsupervised"
os.makedirs(f"{BASE_DIR}/plots", exist_ok=True)
os.makedirs(f"{BASE_DIR}/reports", exist_ok=True)
os.makedirs(f"{BASE_DIR}/objects", exist_ok=True)

def prepare_data():
    """
    Loads data. Returns scaled versions.
    Unsupervised learning is VERY sensitive to scaling.
    """
    try:
        # Check filename here!
        df = pd.read_csv('data/10000_all_params.csv')
    except FileNotFoundError:
        print("ERROR: File '10000_all_params.csv' not found!")
        return None

    df.fillna(0, inplace=True)
    
    # separating label just for visualization comparison later
    # Models will NOT see 'y' during training
    x = df.drop(['Label'], axis=1)
    y = df['Label'] 

    print("Data loaded. Shape:", x.shape)

    # 1. Standard Scaler (Mean=0, Std=1) - Best for K-Means, PCA, DBSCAN, IsoForest
    scaler_std = StandardScaler()
    x_std = pd.DataFrame(scaler_std.fit_transform(x), columns=x.columns)

    # 2. MinMax Scaler (0 to 1) - Required for NMF
    scaler_mm = MinMaxScaler()
    x_mm = pd.DataFrame(scaler_mm.fit_transform(x), columns=x.columns)

    # Note: Removed Binarizer (x_bin) as we deleted FP-Growth

    return x_std, x_mm, y, x.columns

def plot_2d_projection(x_data, labels, model_name, timestamp, method="PCA"):
    """
    Reduces data to 2D to visualize clusters found by the model.
    """
    plt.figure(figsize=(10, 7))
    
    if x_data.shape[1] > 2:
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(x_data)
    else:
        coords = x_data

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame(coords, columns=['Component 1', 'Component 2'])
    plot_df['Cluster'] = labels

    # Plot
    sns.scatterplot(
        x='Component 1', y='Component 2',
        hue='Cluster', 
        palette='viridis', 
        data=plot_df, 
        s=50, alpha=0.6, legend='full'
    )
    plt.title(f'{model_name} Clustering Visualization ({method})')
    plt.savefig(f"{BASE_DIR}/plots/{model_name}_projection_{timestamp}.png", dpi=300)
    plt.close()

def save_report(model_name, metrics, report_text, timestamp):
    path = f"{BASE_DIR}/reports/{model_name}_report_{timestamp}.txt"
    with open(path, "w") as f:
        f.write(f"--- Unsupervised Analysis: {model_name} ---\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("Metrics:\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nDetailed Report:\n")
        f.write(report_text)
    print(f"Report saved: {path}")

def run_clustering(model, x_data, model_name):
    """
    Logic for K-Means, DBSCAN, Hierarchical, IsolationForest
    """
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    print(f"--- Running {model_name} ---")
    
    start_time = time.time()
    
    # Train
    try:
        # IsolationForest uses 'predict', others use 'fit_predict'
        if model_name == "IsolationForest":
            model.fit(x_data)
            labels = model.predict(x_data)
        elif hasattr(model, 'fit_predict'):
            labels = model.fit_predict(x_data)
        else:
            model.fit(x_data)
            labels = model.labels_
    except Exception as e:
        print(f"Error training {model_name}: {e}")
        return

    duration = time.time() - start_time
    
    # Metrics
    if len(x_data) > 20000:
        score_sil = "Skipped (Data too large)"
        score_ch = calinski_harabasz_score(x_data, labels)
    else:
        n_clusters = len(set(labels))
        # Silhouette requires at least 2 clusters and less than N samples
        if 1 < n_clusters < len(x_data):
            score_sil = silhouette_score(x_data, labels)
            score_ch = calinski_harabasz_score(x_data, labels)
        else:
            score_sil = "N/A (1 cluster or noise)"
            score_ch = "N/A"

    unique_labels = np.unique(labels)
    n_clusters_found = len(unique_labels)
    
    # Report Text
    report = f"Clusters found: {n_clusters_found}\n"
    report += f"Labels found (IDs): {unique_labels}\n"
    report += f"Distribution: {pd.Series(labels).value_counts().to_dict()}\n"

    metrics = {
        "Training Time": f"{duration:.2f}s",
        "Silhouette Score": score_sil,
        "Calinski-Harabasz": score_ch,
        "Clusters Count": n_clusters_found
    }

    # Visuals
    plot_2d_projection(x_data, labels, model_name, timestamp)
    save_report(model_name, metrics, report, timestamp)
    
    # Save Model
    joblib.dump(model, f"{BASE_DIR}/objects/{model_name}_{timestamp}.joblib")


def run_dim_reduction(model, x_data, y_true_labels, model_name):
    """
    Logic for PCA, NMF, LLE.
    """
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    print(f"--- Running {model_name} ---")
    
    start_time = time.time()
    
    # Unsupervised transformation
    x_transformed = model.fit_transform(x_data)
        
    duration = time.time() - start_time

    # Align labels with transformed data length (crucial for LLE subset)
    if len(y_true_labels) != len(x_transformed):
        y_labels_plot = y_true_labels.iloc[:len(x_transformed)]
    else:
        y_labels_plot = y_true_labels

    # Plotting the Reduced Dimensions
    plt.figure(figsize=(10, 7))
    if x_transformed.shape[1] >= 2:
        sns.scatterplot(
            x=x_transformed[:,0], 
            y=x_transformed[:,1], 
            hue=y_labels_plot, 
            palette='tab10', 
            s=50
        )
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.title(f"{model_name} Projection (Colored by True Ground Truth)")
        plt.savefig(f"{BASE_DIR}/plots/{model_name}_2D_{timestamp}.png", dpi=300)
        plt.close()

    # Report
    report = f"Output shape: {x_transformed.shape}\n"
    
    if model_name == "PCA":
        report += f"Explained Variance Ratio: {model.explained_variance_ratio_}\n"
        report += f"Total Information Retained: {sum(model.explained_variance_ratio_):.2%}\n"
    elif model_name == "NMF":
        report += f"Reconstruction Error: {model.reconstruction_err_}\n"

    metrics = {"Training Time": f"{duration:.2f}s"}
    save_report(model_name, metrics, report, timestamp)

def main():
    # 1. Prepare Data
    data_pack = prepare_data()
    if data_pack is None: return
    x_std, x_mm, y_true, feature_names = data_pack

    # --- A. CLUSTERING & ANOMALY DETECTION ---
    clustering_models = [
        # 1. K-Means (Fast, simple groups)
        (KMeans(n_clusters=3, random_state=42, n_init=10), x_std, "K-Means"),
        
        # 2. Hierarchical (Detailed structure, memory heavy)
        (AgglomerativeClustering(n_clusters=3), x_std, "Hierarchical_Clustering"),
        
        # 3. DBSCAN (Finds density & noise/anomalies)
        (DBSCAN(eps=3.0, min_samples=10), x_std, "DBSCAN"),

        # 4. Isolation Forest (Best for detecting Attacks/Anomalies)
        (IsolationForest(n_estimators=100, contamination=0.1, random_state=42), x_std, "IsolationForest")
    ]

    for model, data, name in clustering_models:
        run_clustering(model, data, name)

    # --- B. DIMENSIONALITY REDUCTION ---
    dim_red_models = [
        # 1. PCA (Standard visualization)
        (PCA(n_components=2), x_std, "PCA"),
        
        # 2. NMF (Part-based decomposition, non-negative)
        (NMF(n_components=2, init='random', random_state=42, max_iter=500), x_mm, "NMF"),
        
        # 3. LLE (Non-linear manifold, slow - using subset)
        (LocallyLinearEmbedding(n_components=2, n_neighbors=10, method='standard'), x_std.iloc[:2000], "LLE") 
    ]

    for model, data, name in dim_red_models:
        run_dim_reduction(model, data, y_true, name)

    print("\nAll Unsupervised pipelines finished successfully.")

if __name__ == "__main__":
    main()