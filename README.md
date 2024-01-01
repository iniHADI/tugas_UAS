### Laporan Project Machine Learning
# NAMA : Nurul Hadi
# NIM : 211351109
# KELAS : IF MALAM A

### Domain Proyek

Proyek ini bertujuan untuk menganalisis kebiasaan pengguna kartu kredit berdasarkan data transaksi dengan tujuan untuk pengembangan segmentasi pelanggan dalam menentukan strategi pemasaran.

### Business Understanding

Proyek analisis data kartu kredit bertujuan untuk memberikan wawasan mendalam tentang perilaku pengeluaran pelanggan menggunakan kartu kredit,mengeksplorasi pola transaksi dan mendukung pengambilan keputusan yang lebih cerdas dalam konteks pengembangan strategi bisnis yang lebih baik.Bagian laporan ini mencakup :

## Problem Statements

- Bagaimana cara menganalisis data transaksi kartu kredit untuk mengidentifikasi dan segmentasi pelanggan berdasarkan pola pengeluaran mereka, guna memahami preferensi dan perilaku belanja, serta mendukung strategi pemasaran yang lebih terarah.

## Goals

- Meningkatkan pendapatan per pelanggan dengan merancang strategi pemasaran yang didasarkan pada analisis pola pengeluaran untuk meningkatkan frekuensi dan nilai transaksi.

- Meningkatkan kepuasan pelanggan dengan memberikan pelayanan yang lebih baik, memahami preferensi mereka, dan merespons secara proaktif terhadap perubahan pola pengeluaran.

## Solution Statements

1. Meningkatkan efisiensi operasional dengan mengotomatiskan proses deteksi penipuan, manajemen risiko, dan analisis pola pengeluaran, meminimalkan potensi kerugian dan meningkatkan responsibilitas tim operasional.

2. Menganalisis keterkaitan antar kategori pembelian untuk memahami hubungan dan preferensi pelanggan, dengan tujuan meningkatkan strategi penjualan.

3. Pada project ini saya memakai algoritma k-means AffinityPropagation untuk mengelompokkan data dengan memilih kluster sendiri berdasarkan kemiripan antara sampel.Disini saya juga menerapkan KElbowVisualizer untuk membantu memvisualisasikan bagaimana nilai inertia berubah seiring dengan peningkatan jumlah kluster, dan membantu dalam menentukan jumlah kluster.

### Data Understanding

Tahap ini memberikan fondasi analitik untuk sebuah penelitian dengan membuat ringkasan dalam data. Dari data yang telah diambil mengenai dataset yang telah dipakai, terdapat 17 kolom yang berisi 14 kolom float64,dan 3 kolom int64. Disini saya menggunakan algoritma k-means. 

Dataset yang saya gunakan dalam project ini [ccdata](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata/data)
## Variabel-variabel pada ccdata adalah sebagai berikut :

- BALANCE                           :   float64<br>
- BALANCE_FREQUENCY                 :   float64<br>
- PURCHASES                         :   float64<br>
- ONEOFF_PURCHASES                  :   float64<br>
- INSTALLMENTS_PURCHASES            :   float64<br>
- CASH_ADVANCE                      :   float64<br>
- PURCHASES_FREQUENCY               :   float64<br>
- ONEOFF_PURCHASES_FREQUENCY        :   float64<br>
- PURCHASES_INSTALLMENTS_FREQUENCY  :   float64<br>
- CASH_ADVANCE_FREQUENCY            :   float64<br>
- CASH_ADVANCE_TRX                  :   int64<br>
- PURCHASES_TRX                     :   int64<br>
- CREDIT_LIMIT                      :   float64<br>
- PAYMENTS                          :   float64<br>
- MINIMUM_PAYMENTS                  :   float64<br>
- PRC_FULL_PAYMENT                  :   float64<br>
- TENURE                            :   int64 <br>

### Data Preparation
# Data Collection
Untuk data collection ini, saya mendapatkan dataset yang nantinya digunakan dari website kaggle dengan nama dataset ccdata, jika anda tertarik dengan datasetnya, anda bisa click link diatas.

# Data Discovery & Profiling

unggah file dari komputer lokal Anda ke sesi Colab
```
from google.colab import files
files.upload()
```

atur kredensial Kaggle pada notebook
```
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```

unduh dataset dari Kaggle menggunakan perintah kaggle CLI (Command Line Interface).
```
!kaggle datasets download -d arjunbhasin2013/ccdata 
```

buat direktori baru
```
!mkdir ccdata
!unzip ccdata.zip -d ccdata
!ls ccdata 
```

import library yang digunakan
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AffinityPropagation
from yellowbrick.cluster import KElbowVisualizer

import warnings
warnings.filterwarnings("ignore")
```

membaca file CSV 
```
df =pd.read_csv('/content/ccdata/CC GENERAL.csv')
```

data discovery :

```
df.head()
```
![image](<../tugas uas ml/head.1.png>)
```
df.info()
```
![image](<../tugas uas ml/info.1.png>)
```
df.describe().T.style.background_gradient(cmap = 'Spectral')
```
![image](<../tugas uas ml/describe.1.png>)
```
df.shape
```
Exploratory Data Analysis

```
pip install sweetviz
```
![image](<../tugas uas ml/sweetviz.png>)

```
import sweetviz as sv

report = sv.analyze(df)
report.show_notebook()
```
![image](<../tugas uas ml/eda sweetviz.1.png>)
![image](<../tugas uas ml/eda sweetviz.2.png>)
![image](<../tugas uas ml/eda sweetviz.3.png>)
![image](<../tugas uas ml/eda sweetviz.4.png>)
![image](<../tugas uas ml/eda sweetviz.5.png>)

```
sns.regplot(data = df, y = 'CREDIT_LIMIT', x = 'TENURE')
```
![image](<../tugas uas ml/credit limitxtenure.png>)

```
sns.regplot(data = df, y = 'CREDIT_LIMIT', x = 'PAYMENTS')
```
![image](<../tugas uas ml/credit limitxpayment.png>)

```
df.boxplot()
plt.xticks(rotation=90)
plt.show()
```
![image](<../tugas uas ml/boxplot.png>)

```
fig = plt.figure(figsize = (15,15))
sns.heatmap(df.corr(), cmap = 'Blues', square = True, annot = True, linewidths = 0.5) 
```
![image](<../tugas uas ml/heatmap.1.png>)
![image](<../tugas uas ml/heatmap.2.png>)

Data preprocessing

```
df.dropna(inplace=True)
df.isnull().sum()
```
![image](<../tugas uas ml/d.pre.1.png>)

```
df= df.drop(columns=['ONEOFF_PURCHASES', 'PURCHASES_INSTALLMENTS_FREQUENCY','CUST_ID'],axis=1)
df.head()
```
![image](<../tugas uas ml/d.pre.2.png>)

```
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
```

```
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(scaled_df)
    Sum_of_squared_distances.append(km.inertia_)
plt.figure(figsize=(7,5))
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
```
![image](<../tugas uas ml/tes.1.png>)

```
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(1, 15))
elbow.fit(scaled_df)
elbow.show()
```
![image](<../tugas uas ml/tes.2.png>)

```
kmeans = KMeans(n_clusters=4).fit(scaled_df)
kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
scaled_df[0:5]
```
![image](<../tugas uas ml/cluster=4.png>)

```
clusters_kmeans = kmeans.labels_

df["Labels"] = clusters_kmeans

df.head()
```
![image](<../tugas uas ml/labels.png>)

```
df['Labels'].value_counts()
```
![image](<../tugas uas ml/labels value.png>)

### Deployment 
Model yang sudah di buat di deploy menggunakan streamlit: 
Link Aplikasi: [ccdata](https://tugasuas-9rod9a5sun6raunwjrt6zx.streamlit.app/).

![image](<../tugas uas ml/foto apk.png>)
