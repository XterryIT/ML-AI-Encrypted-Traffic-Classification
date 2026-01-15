import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
# --- CONFIGURATION ---
FILE_PATH = 'data/10000_all_params.csv' # Adjust path if needed
SAMPLE_SIZE = 15000  # Number of rows to process

def run_kmeans_analysis():
    # 1. LOAD DATA
    print("--- 1. Loading Data ---")
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"Error: File {FILE_PATH} not found.")
        return

    df.fillna(0, inplace=True)

    
    print("--- 2. Preparing Labels ---")
    y_true = df['Label'].apply(lambda x: 1 if x == 2 else 0).values
    
    # Remove the Label column to get pure features (X)
    X = df.drop(['Label'], axis=1)

    # 3. SCALING (CRITICAL STEP)
    # it dominates the calculation. We must scale everything to mean=0, std=1.
    print("--- 3. Scaling Features ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. K-MEANS CLUSTERING
    print("--- 4. Running K-Means ---")
    # We ask for 2 clusters (assuming one will be Normal, one Attack)
    kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto')
    
    # These are the labels the machine ASSIGNED (0 or 1).
    # Note: The machine does not know that 1 means "Attack". 
    # It just knows it's "Group B".
    cluster_labels = kmeans.fit_predict(X_scaled)

    # 5. EVALUATION
    # ARI Score: 1.0 is perfect match, 0.0 is random guessing.
    # It checks if items that belong together are actually grouped together.
    ari_score = adjusted_rand_score(y_true, cluster_labels)
    print(f"\n>>> Model Accuracy (ARI Score): {ari_score:.4f}")
    
    # Confusion Matrix to see how groups aligned
    print("\nConfusion Matrix (Rows=True, Cols=Predicted):")
    print(confusion_matrix(y_true, cluster_labels))

    # 6. VISUALIZATION (PCA)
    # We have many columns (dimensions). We cannot plot 70 dimensions.
    # We use PCA to compress information into 2 dimensions (X and Y) for plotting.
    print("\n--- 5. Generating Plots (PCA) ---")
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create a DataFrame for Seaborn to plot easily
    plot_df = pd.DataFrame(data=X_pca, columns=['PCA_1', 'PCA_2'])
    plot_df['Cluster_Labels'] = cluster_labels
    plot_df['True_Labels'] = y_true

    # 7. PLOTTING
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # PLOT 1: K-Means Results
    sns.scatterplot(
        x='PCA_1', y='PCA_2', 
        hue='Cluster_Labels', # Coloring by what K-Means found
        data=plot_df, 
        palette='viridis', 
        alpha=0.6, 
        ax=axes[0]
    )
    axes[0].set_title(f'K-Means Clustering Results\n(Machine View)')
    
    # PLOT 2: Real Groups (Ground Truth)
    # We map 0 to "Normal" and 1 to "Attack" for the legend
    plot_df['Label_Name'] = plot_df['True_Labels'].map({0: 'Normal', 1: 'Attack'})
    
    sns.scatterplot(
        x='PCA_1', y='PCA_2', 
        hue='Label_Name', # Coloring by reality
        data=plot_df, 
        palette={'Normal': 'blue', 'Attack': 'red'}, 
        alpha=0.6, 
        ax=axes[1]
    )
    axes[1].set_title('Real Groups (Ground Truth)\n(Reality)')

    plt.tight_layout()
    plt.show()
    print("Done.")

if __name__ == "__main__":
    run_kmeans_analysis()