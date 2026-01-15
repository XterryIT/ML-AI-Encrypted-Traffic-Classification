import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, confusion_matrix

# --- CONFIGURATION ---
FILE_PATH = 'data/10000_chosen_params.csv' 

def load_data(file_path):
    print("--- 1. Loading Data ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None, None

    df.fillna(0, inplace=True)

    print("--- 2. Preparing Labels ---")
    # 0 = Normal, 1 = Attack (Assuming label 2 is Attack in original dataset)
    y_true = df['Label'].apply(lambda x: 1 if x == 2 else 0).values
    
    # Remove the Label column to get pure features (X)
    X = df.drop(['Label'], axis=1)

    print("--- 3. Scaling Features ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_true



def models(X_scaled, y_true):
    print("--- 4. Running K-Means ---")
    

    kmeans = KMeans(n_clusters=6, random_state=42, n_init='auto')
    
    cluster_labels = kmeans.fit_predict(X_scaled)


    ari_score = adjusted_rand_score(y_true, cluster_labels)
    print(f"\n>>> Model Accuracy (ARI Score): {ari_score:.4f}")
    
    print("\nConfusion Matrix (Rows=True, Cols=Predicted):")
    print(confusion_matrix(y_true, cluster_labels))


    return cluster_labels



def visualization(X_scaled, cluster_labels, y_true):
    print("\n--- 5. Generating Plots (PCA) ---")
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Твои цвета (убедись, что ключей столько же, сколько кластеров!)
    my_colors = {
        0: 'red',
        1: 'purple',      
        2: 'green',       
        3: 'blue',       
        4: 'yellow',      
        5: 'black',
    }

    plot_df = pd.DataFrame(data=X_pca, columns=['PCA_1', 'PCA_2'])
    plot_df['Cluster_Labels'] = cluster_labels
    plot_df['True_Labels'] = y_true

    # 7. PLOTTING
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # PLOT 1: Machine View
    sns.scatterplot(
        x='PCA_1', y='PCA_2', 
        hue='Cluster_Labels', 
        data=plot_df, 
        palette=my_colors,
        alpha=0.6, 
        ax=axes[0]
    )
    axes[0].set_title(f'K-Means Clustering Results\n(Machine View)')
    
    # PLOT 2: Reality
    plot_df['Label_Name'] = plot_df['True_Labels'].map({0: 'Normal', 1: 'Attack'})
    
    sns.scatterplot(
        x='PCA_1', y='PCA_2', 
        hue='Label_Name',
        data=plot_df, 
        palette={'Normal': 'blue', 'Attack': 'red'}, 
        alpha=0.6, 
        ax=axes[1]
    )
    axes[1].set_title('Real Groups (Ground Truth)\n(Reality)')

    plt.tight_layout()
    plt.show()
    print("Done.")

def main():

    X_scaled, y_true = load_data(FILE_PATH)
    
    if X_scaled is None:
        return

    cluster_labels = models(X_scaled, y_true)


    visualization(X_scaled, cluster_labels, y_true)

if __name__ == "__main__":
    main()