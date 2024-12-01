# Proyek Pertama Machine Learning Terapan: Predictive Analytics - Miftahus Sholihin

Sumber data: [Lung Cancer-Air Pollution, Alcohol, Smoking & Risk of Lung Cancer](https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link/data)

## Data Understanding

**Deskripsi Dataset**  

Dataset ini berisi informasi tentang pasien kanker paru-paru, termasuk usia, jenis kelamin, paparan polusi udara, konsumsi alkohol, alergi debu, bahaya pekerjaan, risiko genetik, penyakit paru-paru kronis, diet seimbang, obesitas, merokok, perokok pasif, nyeri dada, batuk darah, kelelahan, penurunan berat badan, sesak napas, mengi, kesulitan menelan, kuku jari yang terkatup, dan mendengkur.

| Nama Kolom               | Deskripsi                                              |
| ------------------------ | -------------------------------------------------------|
| Id                       | Id pasien. (object )                                   |
| Age                      | Usia pasien. (int64 )                                  |
| Gender                   | Jenis kelamin pasien. (int64 )                         |
| Air Pollution            | Tingkat paparan polusi udara pada pasien. (int64 )     |
| Alcohol use              | Tingkat penggunaan alkohol oleh pasien. (int64 )       |
| Dust Allergy             | Tingkat alergi debu pada pasien. (int64 )              |
| OccuPational Hazards     | Tingkat bahaya pekerjaan pada pasien. (int64 )         |
| Genetic Risk             | Tingkat risiko genetik pada pasien. (int64 )           |
| chronic Lung Disease     | Tingkat penyakit paru kronis pada pasien. (int64 )     |
| Balanced Diet            | Tingkat pola makan seimbang pasien. (int64 )           |
| Obesity                  | Tingkat obesitas pasien. (int64 )                      |
| Smoking                  | Tingkat merokok oleh pasien. (int64 )                  |
| Passive Smoker           | Tingkat paparan asap rokok pasif pada pasien. (int64 ) |
| Chest Pain               | Tingkat nyeri dada pada pasien. (int64 )               |
| Coughing of Blood        | Tingkat batuk berdarah pada pasien. (int64 )           |
| Fatigue                  | Tingkat kelelahan pada pasien. (int64 )                |
| Weight Loss              | Tingkat penurunan berat badan pada pasien. (int64 )    |
| Shortness of Breath      | Tingkat sesak napas pada pasien. (int64 )              |
| Wheezing                 | Tingkat mengi pada pasien. (int64 )                    |
| Swallowing Difficulty    | Tingkat kesulitan menelan pada pasien. (int64 )        |
| Clubbing of Finger Nails | Tingkat pembengkakan ujung jari pada pasien. (int64 )  |

## Data Loading
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

"""1. Library untuk Pengolahan Data dan Visualisasi:

- numpy dan pandas: Merupakan library dasar untuk operasi numerik dan manipulasi data. pandas sangat berguna untuk menangani data dalam bentuk tabel (DataFrame).
- matplotlib.pyplot dan seaborn: Library untuk visualisasi data. - - matplotlib adalah library dasar untuk plotting, sedangkan seaborn menyediakan fungsi tingkat tinggi untuk membuat visualisasi yang lebih menarik dan informatif.
- warnings: Digunakan untuk menyembunyikan peringatan yang mungkin muncul selama eksekusi kode, agar output menjadi lebih bersih.
2. Library untuk Pembelajaran Mesin (Machine Learning):
Berbagai model klasifikasi dari sklearn (seperti LogisticRegression, KNeighborsClassifier, RandomForestClassifier, DecisionTreeClassifier, GradientBoostingClassifier, GaussianNB): Model-model ini digunakan untuk masalah klasifikasi dalam pembelajaran mesin.
- train_test_split: Membagi dataset menjadi dua subset: data pelatihan dan data pengujian.
- accuracy_score, precision_score, recall_score, f1_score: Ini adalah metrik evaluasi dasar untuk model klasifikasi.
- confusion_matrix: Matriks yang menunjukkan jumlah prediksi yang benar dan salah, yang dapat memberikan wawasan lebih mendalam tentang performa model.
- classification_report: Ringkasan komprehensif yang mencakup precision, recall, dan f1-score untuk setiap kelas.
"""

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/Dicoding/cancer patient data sets.csv')

"""- pd.read_csv(): Fungsi ini digunakan untuk membaca file CSV dan mengonversinya menjadi sebuah objek DataFrame dari library pandas. Sebuah DataFrame adalah struktur data dua dimensi yang mirip dengan tabel, di mana Anda bisa menyimpan dan mengelola data dengan lebih mudah.

- '/content/drive/MyDrive/Dicoding/cancer patient data sets.csv': Ini adalah path lengkap menuju file CSV yang akan dibaca. Path ini menunjukkan bahwa file berada di Google Drive Anda (di dalam folder MyDrive/Dicoding). File ini kemungkinan berisi dataset yang berkaitan dengan pasien kanker.
"""

df.head()

"""Perintah df.head() digunakan untuk menampilkan lima baris pertama dari DataFrame df. Fungsi ini sangat berguna untuk melihat sekilas struktur data dan memastikan bahwa file CSV yang telah dibaca berhasil dimuat dengan benar ke dalam DataFrame."""

df.info()

"""Perintah df.info() digunakan untuk memberikan informasi ringkas tentang DataFrame df. Ini sangat berguna untuk memahami struktur dataset, termasuk jumlah baris, jumlah kolom, nama kolom, tipe data, dan jumlah nilai yang tidak kosong (non-null) dalam setiap kolom."""

df.isna().sum()

"""Perintah df.isna() menghasilkan DataFrame baru yang memiliki nilai True jika data dalam sel tertentu adalah NaN (kosong atau tidak valid), dan False jika tidak. Dengan menambahkan .sum(), pandas menghitung jumlah True (nilai NaN) di setiap kolom. Hasilnya adalah Series yang menunjukkan jumlah nilai kosong untuk setiap kolom dalam DataFrame df."""

df.duplicated().sum()

"""Perintah df.duplicated() menghasilkan sebuah Series yang berisi True untuk setiap baris yang dianggap duplikat dibandingkan dengan baris sebelumnya. Sebuah baris dianggap duplikat jika semua nilai dalam baris tersebut sama persis dengan nilai dalam baris lainnya. Dengan menambahkan .sum(), pandas menghitung jumlah nilai True dalam hasil df.duplicated(), yang menunjukkan jumlah total baris duplikat dalam DataFrame."""

df.drop(columns=['index', 'Patient Id'], axis=1, inplace=True)

"""Menghapus kolom yang tidak relevan: Kolom 'index' dan 'Patient Id' dihapus karena mungkin tidak diperlukan dalam analisis atau model yang sedang dibuat.
Perubahan permanen pada DataFrame: Dengan menggunakan inplace=True, perubahan dilakukan langsung pada DataFrame df tanpa memerlukan penugasan hasil ke variabel baru.
Pembersihan data: Menghapus kolom yang tidak penting dapat membantu membersihkan dataset dan membuatnya lebih fokus pada informasi yang relevan.
"""

df.head()

df.describe()

"""Perintah df.describe() digunakan untuk menghasilkan ringkasan statistik deskriptif dari DataFrame df, yang memberikan gambaran umum mengenai distribusi nilai numerik pada dataset.

## EDA
"""

#check risk level
plt.figure(figsize=(10, 6))
explode = (0, 0, 0.15)
plt.pie(df['Level'].value_counts(),
        labels=df['Level'].unique(),
        explode=explode, autopct='%1.1f%%',
        shadow=True, startangle=90,
        colors=['green', 'yellow', 'red'])
plt.title('Level Distribution')
plt.show()

"""Perintah ini digunakan untuk membuat visualisasi pie chart untuk distribusi tingkat risiko (Level) pada dataset yang ada di kolom 'Level'. Visualisasi ini bertujuan untuk menunjukkan distribusi kategori risiko (misalnya, rendah, sedang, dan tinggi) dalam dataset Anda yang terdapat pada kolom 'Level'. Warna hijau, kuning, dan merah biasanya digunakan untuk menggambarkan kategori risiko yang rendah, sedang, dan tinggi.

### Label Encoding
"""

# Replace "level" with Integer
print('Cancer Levels: ', df['Level'].unique())
df["Level"].replace({'High': 2, 'Medium': 1, 'Low': 0}, inplace=True)
print('Cancer Levels: ', df['Level'].unique())

""">Encode kolom Level agar menjadi numerik dan dapat divisualisasikan"""

df.head()

"""## Data Visualization"""

fig, ax = plt.subplots(ncols=4, nrows=6, figsize=(20, 20))
ax = ax.flatten()

for i, col in enumerate(df.columns):
    sns.regplot(x=col, y='Level', data=df, lowess=True, ax=ax[i])
    ax[i].set_title(col.title())

plt.tight_layout(pad=0.1, w_pad=0.6, h_pad=1)
plt.show()

"""Tujuan dari kode ini adalah untuk mengeksplorasi hubungan antara setiap kolom dalam DataFrame df dengan kolom 'Level' (misalnya level risiko) dengan menggunakan regresi lokal (lowess). Setiap subplot akan menunjukkan hubungan antara kolom numerik lainnya dan kolom 'Level', yang dapat membantu dalam menganalisis bagaimana setiap variabel mempengaruhi atau berkorelasi dengan tingkat risiko yang ada."""

fig, ax = plt.subplots(ncols=4, nrows=6, figsize=(20, 20))
ax = ax.flatten()

columns_to_plot = [col for col in df.columns if col != 'Level']

for i, col in enumerate(columns_to_plot):
    sns.violinplot(x=df['Level'],
                   y=df[col], data=df, hue_order=df['Level'].unique(), palette='Reds', ax=ax[i])
    ax[i].set_title(col.title())

plt.tight_layout(pad=0.1, w_pad=0.2, h_pad=2.5)
plt.show()

"""Visualisasi ini bertujuan untuk menunjukkan distribusi data untuk setiap kolom numerik terhadap level risiko (Level). Violin plot menggabungkan elemen dari box plot dan kernel density estimation (KDE), yang memberi gambaran yang lebih jelas mengenai distribusi, kepadatan data, serta kemungkinan adanya outlier pada setiap level risiko."""

from scipy.stats import norm

fig, ax = plt.subplots(ncols=8, nrows=3, figsize=(24, 12))
ax = ax.flatten()

for i, (column_name, data) in enumerate(df.items()):
    mu, sigma = norm.fit(data)

    sns.histplot(data,
                 kde=True,
                 bins=20,
                 ax=ax[i],
                 label=f'$\mu={mu:.1f}$\n$\sigma={sigma:.1f}$')

    ax[i].set_title(column_name.title())
    ax[i].legend()

plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=1.0)
plt.show()

"""Kode tersebut digunakan untuk membuat histogram dan distribusi kernel density estimation (KDE) untuk setiap kolom dalam DataFrame df. Setiap histogram juga akan menampilkan parameter distribusi normal yang paling cocok (mean μ dan standar deviasi σ) yang diperkirakan dari data menggunakan scipy.stats.norm.fit. Ini akan memberi gambaran tentang seberapa dekat distribusi data dengan distribusi normal."""

correlation_matrix = df.corr()

plt.figure(figsize=(20,15))
sns.heatmap(df.corr(), annot=True, cmap='Reds')
plt.title('Correlation Matrix of All Features', size=20)
plt.show()

"""Kode tersebut digunakan untuk menghitung dan memvisualisasikan matriks korelasi antara fitur-fitur numerik dalam DataFrame df menggunakan heatmap. Heatmap ini akan membantu Anda untuk memahami sejauh mana hubungan antara variabel-variabel dalam dataset.

## Data Preparation
"""

X=df.drop('Level',axis=1)
y=df['Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Kode ini digunakan untuk memisahkan dataset menjadi fitur (X) dan label target (y), lalu membagi dataset tersebut menjadi data pelatihan dan pengujian.

## Modeling
"""

def myModel(X_train, y_train):

    y_train = y_train.values.ravel()

    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)

    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train,y_train)

    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    gb_model = GradientBoostingClassifier()
    gb_model.fit(X_train, y_train)

    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)

    gnb_model = GaussianNB()
    gnb_model = gnb_model.fit(X_train, y_train)

    return lr_model, dt_model, rf_model, gb_model, knn_model, gnb_model

y_test = y_test.values.ravel()

lr_model, dt_model, rf_model, gb_model, knn_model, gnb_model = myModel(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test)
y_pred_gnb = gnb_model.predict(X_test)

"""Kode ini adalah implementasi untuk membangun, melatih, dan menguji beberapa model machine learning untuk tugas klasifikasi.

## Evaluation
"""

def evaluate_models(X_test, y_test, models):
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')

        results.append({'Model': name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall})

    return pd.DataFrame(results)

models = {'Logistic Regression': lr_model, 'Decision Tree': dt_model, 'Random Forest': rf_model, 'Gradient Boosting': gb_model, 'KNN': knn_model, 'Naive Bayes' : gnb_model}

results_df = evaluate_models(X_test, y_test, models)

print(results_df)

"""Kode di atas digunakan untuk mengevaluasi kinerja beberapa model klasifikasi yang telah dilatih sebelumnya menggunakan beberapa metrik evaluasi.

## Hyperparameter Tuning

## Naive Bayes
"""

gnb = GaussianNB()
gnb = gnb.fit(X_train, y_train)
gnb_pred = gnb.predict(X_test)

"""Kode di atas adalah implementasi untuk melatih model Naive Bayes (menggunakan GaussianNB) dan menghasilkan prediksi berdasarkan data pengujian.
- Model dilatih menggunakan data pelatihan (X_train) dan label pelatihan (y_train).
- Metode .fit() adalah standar dalam library scikit-learn untuk melatih model pada data yang diberikan.
- Model yang telah dilatih digunakan untuk memprediksi label untuk data pengujian (X_test).
- Output gnb_pred adalah array prediksi kelas untuk setiap sampel di X_test.
"""

cm = confusion_matrix(y_test, gnb_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test, gnb_pred))

"""### Logistic Regression"""

lr = LogisticRegression()

lr.fit(X_train, y_train)
lr_y_pred = lr.predict(X_test)

"""Kode tersebut adalah implementasi untuk melatih model Logistic Regression dan menghasilkan prediksi untuk data pengujian.
- Model dilatih menggunakan data pelatihan (X_train) dan label pelatihan (y_train).
- Metode .fit() menghitung parameter model (misalnya, bobot dan bias) untuk meminimalkan fungsi loss (biasanya log-loss).
- Model yang telah dilatih digunakan untuk memprediksi kelas untuk data pengujian (X_test).
- Hasil lr_y_pred adalah array prediksi kelas untuk setiap sampel dalam X_test.
"""

cm = confusion_matrix(y_test, lr_y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test, lr_y_pred))

"""### Decision Tree"""

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)
dt_y_pred = dt.predict(X_test)

"""
Kode ini melatih model Decision Tree Classifier dan menghasilkan prediksi pada data pengujian.
- Membuat instance model Decision Tree Classifier.
- random_state=42: Menjamin replikasi hasil dengan menetapkan nilai acak yang konsisten.
- Model dilatih menggunakan data pelatihan (X_train) dan label pelatihan (y_train).
- Model mempelajari aturan pemisahan berdasarkan fitur untuk memaksimalkan akurasi prediksi.
- Menggunakan model yang telah dilatih untuk memprediksi kelas pada data pengujian (X_test).
- Hasil prediksi disimpan di dt_y_pred sebagai array yang merepresentasikan kelas yang diprediksi untuk setiap sampel."""

cm = confusion_matrix(y_test, dt_y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test, dt_y_pred))

"""## Decision Tree Plot"""

from sklearn.tree import plot_tree

plt.figure(figsize=(25, 20))
plot_tree(dt, feature_names = df.columns, filled = True, rounded = False)
plt.title('Decision Tree Plot', fontsize=20)

plt.show()

"""### Random Forest"""

rf = RandomForestClassifier(random_state=123)

rf.fit(X_train, y_train)
rf_y_pred = rf.predict(X_test)

"""
Kode ini melatih model Random Forest Classifier dan menghasilkan prediksi pada data pengujian.
- Membuat instance model Random Forest Classifier.
- random_state=123: Menjamin hasil eksperimen konsisten dengan menetapkan nilai acak yang tetap.
- Melatih model menggunakan data pelatihan (X_train) dan labelnya (y_train).
- Model yang telah dilatih digunakan untuk memprediksi kelas pada data pengujian (X_test).
- Prediksi kelas disimpan dalam array rf_y_pred, berisi kelas yang diprediksi untuk setiap sampel."""

cm = confusion_matrix(y_test, rf_y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test, rf_y_pred))

"""### Gradient Boosting"""

gb = GradientBoostingClassifier(random_state=123)

gb.fit(X_train, y_train)

gb_y_pred = gb.predict(X_test)

"""
Kode ini melatih model Gradient Boosting Classifier dan menghasilkan prediksi untuk data pengujian.
- Model dilatih menggunakan data pelatihan X_train dan labelnya y_train.
- Gradient Boosting mengoptimalkan fungsi loss (misalnya log-loss untuk klasifikasi) dengan cara iteratif:
  - Membuat pohon keputusan (decision trees) kecil secara bertahap.
  - Setiap pohon baru difokuskan untuk mengurangi kesalahan residual dari model sebelumnya.
- Model yang sudah dilatih digunakan untuk memprediksi kelas pada data pengujian X_test.
- Prediksi disimpan dalam array gb_y_pred, berisi prediksi kelas untuk setiap sampel dalam data pengujian."""

cm = confusion_matrix(y_test, gb_y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test, gb_y_pred))

"""### KNN (k-nearest neighbors)"""

knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
knn.fit(X_train, y_train)
knn_y_pred = knn.predict(X_test)

"""
Kode ini melatih model K-Nearest Neighbors (KNN) dengan konfigurasi khusus dan menghasilkan prediksi untuk data pengujian.
- Membuat instance dari model K-Nearest Neighbors (KNN).
- Parameter:
  - n_neighbors=7: Menggunakan 7 tetangga terdekat untuk membuat keputusan klasifikasi.
    - Artinya, untuk setiap sampel, algoritma melihat 7 data terdekat dalam ruang fitur untuk menentukan kelas.
  - weights='distance': Memberikan bobot berdasarkan jarak. Tetangga yang lebih dekat memiliki bobot lebih besar dalam memengaruhi keputusan klasifikasi dibandingkan yang lebih jauh.
- Model dilatih menggunakan data pelatihan X_train dan labelnya y_train.
- Model menggunakan data pengujian X_test untuk memprediksi kelas.
- Prediksi disimpan dalam array knn_y_pred, yang berisi kelas untuk setiap sampel dalam X_test."""

cm = confusion_matrix(y_test, knn_y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'HIgh'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_test, knn_y_pred))
