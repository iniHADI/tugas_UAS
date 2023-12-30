import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Membaca dataset dari file CSV
df =pd.read_csv('/content/ccdata/CC GENERAL.csv')

df = pd.DataFrame(df)

# Memilih kolom yang akan digunakan untuk clustering
selected_columns = ['Labels']
selected_data = df[selected_columns]

# Menentukan jumlah cluster maksimum yang akan diuji
max_clusters = 10

# Melakukan iterasi untuk berbagai jumlah cluster dan menghitung inersia
inertias = []
for k in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(selected_data)
    inertias.append(kmeans.inertia_)

# Memplot grafik elbow
plt.plot(range(1, max_clusters + 1), inertias, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Cluster Number')
plt.show()