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
    Funkcja oblicza średnie wartości parametrów dla każdego klastra
    i przywraca je do oryginalnych jednostek (np. bajty, sekundy).
    """
    print(f"\n--- Interpretacja Klastrów (Centroide) dla {model_name} ---")
    
    # Tworzymy tymczasowy DataFrame ze skalowanymi danymi
    df_temp = pd.DataFrame(X_scaled, columns=feature_names)
    df_temp['Cluster'] = cluster_labels
    
    # 1. Obliczamy średnie (centroidy) dla każdego klastra
    # (Wciąż w skali standardowej)
    means_scaled = df_temp.groupby('Cluster').mean()
    
    # 2. Odwracamy skalowanie (Inverse Transform)
    # Żeby zobaczyć prawdziwe liczby (np. Port 80, a nie 0.54)
    try:
        means_original = scaler.inverse_transform(means_scaled)
        
        # Tworzymy czytelną tabelę
        df_means = pd.DataFrame(means_original, columns=feature_names, index=means_scaled.index)
        
        # Transponujemy (.T), żeby klastry były kolumnami (łatwiej czytać wiele parametrów)
        report_df = df_means.T
        
        # Wyświetlamy pierwsze 5 parametrów w konsoli dla podglądu
        print("\n[TOP 5 PARAMETRÓW - ŚREDNIE WARTOŚCI (Oryginalne Jednostki)]")
        print(report_df.head(5))
        
        # Zapisujemy pełną tabelę do CSV, bo parametrów jest dużo
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        csv_path = f"{OUTPUT_DIR}/{safe_name}_interpretation.csv"
        report_df.to_csv(csv_path)
        print(f"✅ Pełna tabela średnich zapisana do: {csv_path}")
        
    except Exception as e:
        print(f"⚠️ Nie można zinterpretować klastrów (np. dla LOF/DBSCAN z samym szumem): {e}")

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

    print(f"\n{'ID':<5} | {'Rozmiar':<8} | {'% Ataków':<10} | {'WERDYKT'}")
    print("-" * 65)

    for cluster_id, row in stats.iterrows():
        percent = row['mean'] * 100
        if percent > 95: verdict = "🔴 CZYSTY ATAK"
        elif percent > 50: verdict = "🟠 MIESZANY (Wysokie Ryzyko)"
        elif percent > 5: verdict = "🔵 MIESZANY (Szum)"
        else: verdict = "🟢 CZYSTA NORMA"
        print(f"{cluster_id:<5} | {int(row['count']):<8} | {percent:>6.1f}%   | {verdict}")

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
    print(f"✅ Zapisano wykres do: {file_path}")

    print(f"👀 Wyświetlanie wykresu... (Zamknij okno, aby kontynuować)")
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
            
    print(f"\n🎉 Gotowe! Sprawdź folder: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()