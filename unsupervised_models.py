import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- IMPORTY ---
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, confusion_matrix

# --- KONFIGURACJA ---
FILE_PATH = 'data/10000_chosen_params.csv'
OUTPUT_DIR = "models/unsupervised"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(file_path):
    print("--- 1. Wczytywanie Danych ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Błąd: Plik {file_path} nie znaleziony.")
        return None, None, None, None

    df.fillna(0, inplace=True)

    # 0 = Norma, 1 = Atak
    y_true = df['Label'].apply(lambda x: 1 if x == 2 else 0).values
    X = df.drop(['Label'], axis=1)
    feature_names = X.columns  # Zapisujemy nazwy kolumn

    print("--- 3. Skalowanie Cech ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Zwracamy scaler i nazwy cech, aby móc później odwrócić transformację
    return X_scaled, y_true, scaler, feature_names

def get_models_dict():
    return {
        "K-Means": KMeans(n_clusters=6, random_state=42, n_init='auto'),
        "Hierarchical": AgglomerativeClustering(n_clusters=6),
        "GMM_Distribution": GaussianMixture(n_components=6, random_state=42),
        "DBSCAN": DBSCAN(eps=3.0, min_samples=10),
        "LOF": LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    }

def interpret_clusters(X_scaled, cluster_labels, scaler, feature_names, model_name):
    """
    Analizuje, które parametry są najważniejsze (dominujące) dla każdego klastra.
    Sortuje parametry według odchylenia od normy.
    """
    print(f"\n--- Interpretacja Cech Dominujących dla {model_name} ---")
    
    # 1. Tworzymy DataFrame ze skalowanymi danymi (To są "Z-Scores")
    df_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    df_scaled['Cluster'] = cluster_labels
    
    # 2. Obliczamy średnie dla klastrów (w skali standardowej)
    # Jeśli wartość wynosi 0, oznacza to "średnią globalną".
    # Im dalej od 0 (np. 5.0 lub -3.0), tym ważniejsza jest cecha.
    means_scaled = df_scaled.groupby('Cluster').mean()
    
    # 3. Odwracamy skalowanie, żeby pokazać użytkownikowi prawdziwe liczby w nawiasie
    means_original = pd.DataFrame(
        scaler.inverse_transform(means_scaled), 
        columns=feature_names, 
        index=means_scaled.index
    )

    print("\n[TOP 5 CECH DOMINUJĄCYCH DLA KAŻDEGO KLASTRA]")
    print("(Wartość skazująca: Siła wpływu cechy. Prawdziwa wartość w nawiasie)")
    print("-" * 60)

    # Przechodzimy przez każdy klaster
    for cluster_id in means_scaled.index:
        print(f"\n>>> Klaster {cluster_id}:")
        
        # Sortujemy cechy według wartości bezwzględnej w skali standardowej
        # (Największe odchylenie od 0 = Najważniejsza cecha)
        cluster_row = means_scaled.loc[cluster_id]
        
        # Bierzemy 5 najważniejszych cech
        top_features = cluster_row.abs().sort_values(ascending=False).head(5)
        
        for feature_name, impact_score in top_features.items():
            # impact_score - jak bardzo cecha wyróżnia się na tle innych (skalowana)
            # real_value - prawdziwa wartość (np. liczba bajtów)
            real_value = means_original.loc[cluster_id, feature_name]
            
            # Określamy kierunek (Powyżej czy Poniżej średniej)
            direction = "WYSOKI" if cluster_row[feature_name] > 0 else "NISKI"
            
            print(f"   • {feature_name:<25} | Wpływ: {impact_score:>5.2f} ({direction}) | Średnia: {real_value:.2f}")

    # Zapis pełnego raportu do CSV (Opcjonalnie)
    safe_name = model_name.replace(" ", "_")
    csv_path = f"{OUTPUT_DIR}/{safe_name}_features.csv"
    means_original.T.to_csv(csv_path)
    print(f"\nPełna tabela średnich zapisana do: {csv_path}")

def analyze_clusters(cluster_labels, y_true, model_name):
    print(f"\n--- Analiza dla: {model_name} ---")
    
    # 1. Obliczenie Macierzy Pomyłek (Confusion Matrix)
    cm = confusion_matrix(y_true, cluster_labels)
    
    print("\n[MACIERZ POMYŁEK]")
    print("Wiersze: Klasy rzeczywiste (0=Norma, 1=Atak)")
    print("Kolumny: Numery klastrów")
    print(cm)

    # 2. Szczegółowa analiza (Profilowanie)
    df_analysis = pd.DataFrame({'Cluster': cluster_labels, 'Is_Attack': y_true})
    stats = df_analysis.groupby('Cluster')['Is_Attack'].agg(['count', 'mean'])


def visualization(X_scaled, cluster_labels, y_true, model_name):
    print(f"--- Przetwarzanie wykresu dla {model_name} ---")
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    my_colors = {
        -1: 'black', 0: 'red', 1: 'purple', 2: 'green',       
        3: 'blue', 4: 'yellow', 5: 'orange', 6: 'cyan', 7: 'magenta'
    }

    plot_df = pd.DataFrame(data=X_pca, columns=['PCA_1', 'PCA_2'])
    plot_df['Cluster_Labels'] = cluster_labels
    plot_df['True_Labels'] = y_true

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # WYKRES 1: Widok Maszynowy
    unique_labels = set(cluster_labels)
    palette_to_use = my_colors if all(l in my_colors for l in unique_labels) else 'tab10'

    sns.scatterplot(
        x='PCA_1', y='PCA_2', hue='Cluster_Labels', data=plot_df, 
        palette=palette_to_use, alpha=0.6, ax=axes[0]
    )
    axes[0].set_title(f'Wyniki {model_name} (Widok Maszynowy)')
    
    # WYKRES 2: Rzeczywistość
    plot_df['Label_Name'] = plot_df['True_Labels'].map({0: 'Norma', 1: 'Atak'})
    
    sns.scatterplot(
        x='PCA_1', y='PCA_2', hue='Label_Name', data=plot_df, 
        palette={'Norma': 'blue', 'Atak': 'red'}, alpha=0.6, ax=axes[1]
    )
    axes[1].set_title('Prawda Podstawowa (Rzeczywistość)')

    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    file_path = f"{OUTPUT_DIR}/{safe_name}.png"
    
    plt.tight_layout()
    plt.savefig(file_path)
    print(f"Zapisano wykres do: {file_path}")

    print(f"Wyświetlanie wykresu... (Zamknij okno, aby kontynuować)")
    plt.show() 
    plt.close()

def main():
    # Zmienione: odbieramy 4 wartości
    X_scaled, y_true, scaler, feature_names = load_data(FILE_PATH)
    
    if X_scaled is None: return

    models = get_models_dict()

    for name, model in models.items():
        print(f"\n>>> Uruchamianie {name}...")
        try:
            if name == "GMM_Distribution":
                model.fit(X_scaled)
                labels = model.predict(X_scaled)
            elif name == "LOF":
                labels = model.fit_predict(X_scaled)
            else:
                labels = model.fit_predict(X_scaled)

            # 1. Analiza składu (Atak vs Norma)
            analyze_clusters(labels, y_true, name)
            
            # 2. NOWOŚĆ: Interpretacja parametrów (Co to za ruch?)
            interpret_clusters(X_scaled, labels, scaler, feature_names, name)
            
            # 3. Wizualizacja
            visualization(X_scaled, labels, y_true, name)

        except Exception as e:
            print(f"Nie udało się uruchomić {name}: {e}")
            
    print(f"\nGotowe! Sprawdź folder: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()