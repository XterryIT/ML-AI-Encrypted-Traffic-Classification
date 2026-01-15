import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- IMPORTS FOR NEW ALGORITHMS ---
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

# --- CONFIGURATION ---
FILE_PATH = 'data/10000_chosen_params.csv'
OUTPUT_DIR = "models/unsupervised"

# Create directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(file_path):
    print("--- 1. Loading Data ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None, None

    df.fillna(0, inplace=True)

    # 0 = Normal, 1 = Attack
    y_true = df['Label'].apply(lambda x: 1 if x == 2 else 0).values
    X = df.drop(['Label'], axis=1)

    print("--- 3. Scaling Features ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_true

def get_models_dict():
    return {
        "K-Means": KMeans(n_clusters=6, random_state=42, n_init='auto'),
        "Hierarchical": AgglomerativeClustering(n_clusters=6),
        "GMM_Distribution": GaussianMixture(n_components=6, random_state=42),
        "DBSCAN": DBSCAN(eps=3.0, min_samples=10),
        "LOF": LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    }


def visualization(X_scaled, cluster_labels, y_true, model_name):
    print(f"--- Processing Plot for {model_name} ---")
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Extended colors
    my_colors = {
        -1: 'black', 0: 'red', 1: 'purple', 2: 'green',       
        3: 'blue', 4: 'yellow', 5: 'orange', 6: 'cyan', 7: 'magenta'
    }

    plot_df = pd.DataFrame(data=X_pca, columns=['PCA_1', 'PCA_2'])
    plot_df['Cluster_Labels'] = cluster_labels
    plot_df['True_Labels'] = y_true

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # PLOT 1: Machine View
    unique_labels = set(cluster_labels)
    palette_to_use = my_colors if all(l in my_colors for l in unique_labels) else 'tab10'

    sns.scatterplot(
        x='PCA_1', y='PCA_2', hue='Cluster_Labels', data=plot_df, 
        palette=palette_to_use, alpha=0.6, ax=axes[0]
    )
    axes[0].set_title(f'{model_name} Results (Machine View)')
    
    # PLOT 2: Reality
    plot_df['Label_Name'] = plot_df['True_Labels'].map({0: 'Normal', 1: 'Attack'})
    sns.scatterplot(
        x='PCA_1', y='PCA_2', hue='Label_Name', data=plot_df, 
        palette={'Normal': 'blue', 'Attack': 'red'}, alpha=0.6, ax=axes[1]
    )
    axes[1].set_title('Ground Truth (Reality)')

    # --- SAVE FILE FIRST ---
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    file_path = f"{OUTPUT_DIR}/{safe_name}.png"
    
    plt.tight_layout()
    plt.savefig(file_path)
    print(f"✅ Saved to: {file_path}")

    # --- SHOW WINDOW SECOND ---
    plt.show() 
    
    # Clear memory
    plt.close()

def main():
    X_scaled, y_true = load_data(FILE_PATH)
    if X_scaled is None: return

    models = get_models_dict()

    for name, model in models.items():
        print(f"\n>>> Running {name}...")
        try:
            if name == "GMM_Distribution":
                model.fit(X_scaled)
                labels = model.predict(X_scaled)
            elif name == "LOF":
                labels = model.fit_predict(X_scaled)
            else:
                labels = model.fit_predict(X_scaled)

            visualization(X_scaled, labels, y_true, name)

        except Exception as e:
            print(f"Failed to run {name}: {e}")
            

if __name__ == "__main__":
    main()