import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score

# Fungsi untuk menjalankan KMeans clustering dan mendapatkan titik pusat cluster
def run_kmeans(n_clusters, X):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)

    return labels, cluster_centers

# Fungsi untuk menampilkan scatter plot hasil clustering
def show_cluster_scatterplot(X, labels, cluster_centers):
    fig, ax = plt.subplots(figsize=(10, 8))
    scatterplot = plt.scatter(x=X[X.columns[0]], y=X[X.columns[1]], c=labels, cmap='hsv', marker='o', s=30, edgecolor='k')
    plt.legend(*scatterplot.legend_elements(), title='Clusters', loc='upper right', bbox_to_anchor=(1.2, 1))

    for label in np.unique(labels):
        cluster_mean = cluster_centers.loc[label, [X.columns[0], X.columns[1]]]
        plt.annotate(label,
                     cluster_mean,
                     ha='center',
                     va='center',
                     color='black',
                     size=10,
                     weight='bold',
                     backgroundcolor='white',
                     bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

    plt.title('Cluster Analysis')
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])

    st.pyplot(fig)

# Fungsi untuk menampilkan metode "elbow"
def show_elbow_method(X, max_clusters=10):
    distortions = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion (Inertia)')
    
    st.pyplot(plt.gcf())  # Clear the figure explicitly

# Fungsi untuk menampilkan silhouette scores
def show_silhouette_scores(X, max_clusters=10):
    silhouette_scores = []

    for num_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, labels)
        silhouette_scores.append(silhouette_avg)

    # Menampilkan silhouette scores dalam tabel
    st.subheader("Silhouette Scores for Different Numbers of Clusters:")
    scores_df = pd.DataFrame({'Number of Clusters': list(range(2, max_clusters + 1)), 'Silhouette Score': silhouette_scores})
    st.table(scores_df)

    # Menampilkan plot silhouette scores
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Different Numbers of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    
    st.pyplot(plt.gcf())  # Clear the figure explicitly

# Aplikasi Streamlit
def main():
    st.title("Visualisasi KMeans Clustering")

    # Sidebar
    st.sidebar.header("Pengaturan")
    n_clusters = st.sidebar.slider("Pilih Jumlah Cluster", 2, 10, value=4)

    # Generate a sample dataset
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
    X = pd.DataFrame(X, columns=['PAYMENTS', 'CREDIT_LIMIT'])

    # Menampilkan dataset secara menyeluruh
    st.subheader("Dataset:")
    st.write(X)

    # Jalankan KMeans untuk mendapatkan label dan titik pusat cluster
    labels, cluster_centers = run_kmeans(n_clusters, X)

    # Menampilkan dataset asli dengan label cluster yang ditetapkan
    original_data = pd.DataFrame(X, columns=['PAYMENTS', 'CREDIT_LIMIT'])
    original_data['Cluster'] = labels
    st.subheader("Dataset Asli dengan Label Cluster:")
    st.write(original_data)

    # Menampilkan scatter plot hasil clustering
    show_cluster_scatterplot(X, labels, cluster_centers)

    # Menampilkan metode "elbow"
    show_elbow_method(X)

    # Menampilkan silhouette scores
    show_silhouette_scores(X)

if __name__ == "__main__":
    main()
