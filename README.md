## Laporan Proyek Machine Learning - Miftahus Sholihin
Data yang digunakan pada proyek ini berasal dari : https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link/data

## Domain Proyek
Kanker paru-paru adalah salah satu penyebab utama kematian di seluruh dunia. Faktor risiko seperti merokok, paparan zat berbahaya, dan gaya hidup dapat meningkatkan kemungkinan seseorang terkena kanker paru-paru. Deteksi dini melalui prediksi berbasis data memungkinkan pengobatan lebih cepat dan peluang kesembuhan yang lebih tinggi. Proyek ini bertujuan mengembangkan model prediksi berbasis machine learning yang dapat membantu identifikasi individu berisiko tinggi, sehingga dapat diterapkan dalam skrining kesehatan atau program pencegahan.

## Problem Statemen
- Bagaimana membangun model machine learning (Naive Bayes, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, dan K-Nearest Neighbors (KNN)) untuk mendeteksi kanker paru-paru pada pasisen?
- Bagaimana tingkat akurasi dari model machine learning (Naive Bayes, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, dan K-Nearest Neighbors (KNN)) untuk deteksi kanker paru-paru.

## Goals
- Membangun model machine learning yang dapat mengidentifikasi pasien dengan risiko tinggi terkena kanker paru-paru menggunakan data medis.
- Mendapatakn algoritma machine learning terbaik untuk deteksi kanker paru-paru.

## Solution statements
- Membangun enam algoritma klasifikasi yaitu Naive Bayes, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, dan K-Nearest Neighbor.

## Data Understanding
Dataset ini berisi informasi tentang pasien kanker paru-paru, termasuk usia, jenis kelamin, paparan polusi udara, konsumsi alkohol, alergi debu, bahaya pekerjaan, risiko genetik, penyakit paru-paru kronis, diet seimbang, obesitas, merokok, perokok pasif, nyeri dada, batuk darah, kelelahan, penurunan berat badan, sesak napas, mengi, kesulitan menelan, kuku jari yang terkatup, dan mendengkur. 

Dataset ini berisi informasi medis sejumlah 1000 pasien yang mencakup 20 variabel kategorikal. Dataset ini diambil dari link https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link/data. Dimana dalam dataset tersebut tidak ada duplikasi data dan juga tidak ada data yang kosong atau missing value. Dataset ini terdiri dari 26 kolom.

## Deskripsi Variabel
| Nama Kolom               | Deskripsi                                                   |
| ------------------------ | ----------------------------------------------------------- |
| Id                       | Id pasien(object )                                      |
| Age                      | Usia pasien(int64 )                                      |
| Gender                   | Jenis kelamin pasien (int64 )                         |
| Air Pollution            | Tingkat paparan polusi udara pada pasien(int64 )     |
| Alcohol use              | Tingkat penggunaan alkohol oleh pasien(int64 )       |
| Dust Allergy             | Tingkat alergi debu pada pasien(int64 )              |
| OccuPational Hazards     | Tingkat bahaya pekerjaan pada pasien(int64 )         |
| Genetic Risk             | Tingkat risiko genetik pada pasien(int64 )           |
| chronic Lung Disease     | Tingkat penyakit paru kronis pada pasien(int64 )     |
| Balanced Diet            | Tingkat pola makan seimbang pasien(int64 )           |
| Obesity                  | Tingkat obesitas pasien(int64 )                      |
| Smoking                  | Tingkat merokok oleh pasien(int64 )                  |
| Passive Smoker           | Tingkat paparan asap rokok pasif pada pasien(int64 ) |
| Chest Pain               | Tingkat nyeri dada pada pasien(int64 )               |
| Coughing of Blood        | Tingkat batuk berdarah pada pasien(int64 )           |
| Fatigue                  | Tingkat kelelahan pada pasien(int64 )                |
| Weight Loss              | Tingkat penurunan berat badan pada pasien(int64 )    |
| Shortness of Breath      | Tingkat sesak napas pada pasien(int64 )              |
| Wheezing                 | Tingkat mengi pada pasien(int64 )                    |
| Swallowing Difficulty    | Tingkat kesulitan menelan pada pasien(int64 )        |
| Clubbing of Finger Nails | Tingkat pembengkakan ujung jari pada pasien(int64 )  |

## Data Visualization
Pada proyek ini beberapa teknik visualisasi data digunakan, seperti pie chart, reggression plot, violin plot, histogram distribusi normal, dan heatmap matriks korelasi untuk memahami data.
1. Pie Chart: Status Distribution of Dataset. Visualisasi ini digunakan untuk menunjukkan distribusi kategori pada variabel target Level dalam bentuk persentase. Pie chart memberikan gambaran yang jelas tentang proporsi kategori dalam dataset, seperti Low, Medium, dan High.

    <details>
        <summary>Lihat gambar</summary>            
            ![Pie chart](https://github.com/aziza8/MLP/blob/main/piechart.png?raw=true)
    </details>

    Insight:
    - Distribusi tidak merata di antara tiga kategori risiko.
    - Risiko Tinggi (36.5%), Sedang (33.2%), dan Rendah (30.3%).

2. Regplot: Relationship between Features and Target. Untuk setiap fitur dalam dataset, dibuat regression plot terhadap variabel target Level. Regplot ini menunjukkan hubungan linier antara fitur dan target, dengan tambahan Lowess smoothing untuk menampilkan tren yang lebih jelas. Visualisasi ini membantu dalam memahami pola antara fitur dan variabel target.
    <details>
        <summary>Lihat gambar</summary>
            ![Regplot](https://github.com/aziza8/MLP/blob/main/regplot.png?raw=true)
    </details>
    Insight:
     - Hubungan antara fitur dan target bervariasi. Beberapa menunjukkan korelasi yang kuat (positif maupun negatif).
    - Terdapat fitur-fitur yang memiliki hubungan lebih kuat dan lebih jelas dengan target.

3. Violin Plot: Distribution of Features by Level. Violin plot digunakan untuk menggambarkan distribusi setiap fitur numerik berdasarkan kategori level pada variabel target Level. Plot ini memberikan informasi tentang distribusi dan kepadatan data untuk masing-masing kategori (Low, Medium, High), serta memberikan perbandingan visual yang kuat antar kategori.

    <details>
        <summary>Lihat gambar</summary>
            ![Violin](https://github.com/aziza8/MLP/blob/main/violinplot.png?raw=true)
    </details>
    Insight:

    - Bentuk dan lebar violin plot yang berbeda-beda untuk setiap kategori menunjukkan bahwa distribusi fitur bervariasi secara signifikan antar kelompok 'Level'. Beberapa fitur menunjukkan perbedaan yang jelas antara kelompok 'Level'. Misalnya, untuk fitur 'Usia' pada kelompok 'Tinggi' cenderung lebih tinggi dibandingkan kelompok 'Rendah', maka dapat disimpulkan bahwa usia lebih tinggi terkait dengan tingkat risiko yang lebih tinggi.
    - Garis tengah pada violin plot mewakili median, sedangkan bagian yang lebih tebal menunjukkan rentang interkuartil (IQR). Ini memberikan gambaran tentang lokasi pusat data dan seberapa tersebar data di sekitar median.
    - Titik-titik data yang berada di luar 'kumis' violin plot dianggap sebagai outlier.

4. Histogram with Normal Distribution Fit. Histogram dibuat untuk setiap fitur dalam dataset untuk mengevaluasi distribusi datanya. Selain itu, estimasi mean (μ) dan standard deviation (σ) dari distribusi ditambahkan, bersama dengan KDE plot untuk melihat kepadatan distribusi. Visualisasi ini membantu dalam memahami apakah fitur mengikuti distribusi normal atau tidak.

    <details>
        <summary>Lihat gambar</summary>
            ![Histogram](https://github.com/aziza8/MLP/blob/main/histplot.png?raw=true)
    </details>
    Insight:

    - Jika data mengikuti distribusi normal, maka sebagian besar data akan terkonsentrasi di sekitar rata-rata (mean), dan semakin menjauh dari rata-rata, frekuensinya akan semakin menurun.
    - Jika histogram miring ke kanan, banyak data berkumpul di nilai yang lebih rendah. Jika miring ke kiri, banyak data berkumpul di nilai yang lebih tinggi.
    - Kurtosis yang mengukur seberapa 'runcing' atau 'gepeng' distribusi. Distribusi leptokurtik (runcing), sedangkan distribusi platykurtik (gepeng).

5. Heatmap: Correlation Matrix of Features. Heatmap digunakan untuk memvisualisasikan matriks korelasi antar fitur dalam dataset. Matriks korelasi ini menunjukkan hubungan linear antara fitur, dengan nilai korelasi ditampilkan secara numerik di dalam plot. Heatmap ini sangat berguna untuk mengidentifikasi fitur-fitur yang berkorelasi kuat satu sama lain, yang bisa membantu dalam proses feature selection.

    <details>
        <summary>Lihat gambar</summary>
            ![Heatmap](https://github.com/aziza8/MLP/blob/main/heatmap.png?raw=true)
    </details>
    Insight:

    - Banyak kotak berwarna merah pekat menandakan tingginya korelasi antar variabel.
    - Variabel 'Air Pollution', 'Alcohol Use', 'Dust Allergy', 'Occupational Hazards', 'Genetic Risk', dan 'Chronic Lung Disease' menunjukkan bahwa paparan polusi udara, konsumsi alkohol, alergi debu, bahaya pekerjaan, risiko genetik, dan penyakit paru-paru kronis saling mempengaruhi.

## Data Preparation
Data preparation diperlukan untuk memastikan data yang bersih dan siap digunakan oleh model machine learning. Pada proyek ini terdapat bebera langkah untuk data preparation:
1. Penyesuaian Data: Mengatur kolom `index` pada dataset supaya tidak termasuk ke dalam variabel.
2. Pembersihan Data: Menghapus kolom `'Patient Id'` yang tidak relevan untuk model prediksi.
3. Encoding: Menggunakan teknik label encoding untuk mengubah variabel `'Level'` menjadi variabel numerik.
4. Pemisahan Fitur dan Label: Memisahkan fitur (X) yang merupakan variabel independen dan label (y) yang merupakan variabel dependen untuk model.
5. Train test split: Membagi data menjadi data training (80%) dan data testing (20%).

Proyek ini index dan Patient Id dilakukan penghapusan. Hal ini dilakukan karena kedua kolom tersebut tidak berpengaruh terhadap proses selanjutnya.

## Modeling
Pada proyek ini, beberapa model machine learning digunakan untuk proses klasifikasi kanker paru-paru. Model-model tersebut adalah:

1. Naive Bayes,
2. Logistic Regression
3. Decision Tree
4. Random Forest
5. Gradient Boosting
6. K-Nearest Neighbors (KNN)

Model yang sudah dibuat kemudian dilakukan proses pelatihan. Berikut adalah code untuk proses pelatihan dari model yang dibuat.

```python
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
```

Model yang dibuat menggunakan **default parameter**.
1. Gaussian Naive Bayes adalah varian dari Naive Bayes yang digunakan untuk fitur kontinu. Algoritma ini mengasumsikan bahwa distribusi setiap fitur mengikuti distribusi normal (Gaussian).
2. Logistic Regression (LR). Model ini menggunakan sigmoid function untuk mengubah keluaran linier menjadi probabilitas. Model ini sederhana dan cepat, namun mungkin tidak terlalu kuat pada dataset yang lebih kompleks.

   Parameter default:

   - `penalty='l2'` Penalti regulasi L2 diterapkan untuk mencegah overfitting.
   - `C=1.0` Parameter inverse regularization strength. Artinya semakin kecil nilai C, semakin kuat regulasi diterapkan.
   - `solver='lbfgs'` Solver digunakan untuk optimasi, dan `'lbfgs'` sangat cocok untuk dataset ukuran kecil hingga menengah.
   - `random_state` Default-nya adalah None, artinya tidak ada seed yang diterapkan.

3. Decision Tree (DT). Algoritma ini bekerja dengan memisahkan data berdasarkan aturan keputusan yang dihasilkan dari fitur dataset. Setiap node dalam tree mewakili keputusan berbasis satu fitur. Decision Tree cenderung mudah diinterpretasikan namun rentan terhadap overfitting jika tidak dipangkas dengan baik.

   Parameter default:

   - `criterion='gini'` Menggunakan Gini impurity untuk mengukur kualitas split.
   - `max_depth=None` Tidak ada batasan pada kedalaman pohon, yang memungkinkan pohon tumbuh sampai sempurna.
   - `random_state` Default-nya adalah None, artinya tidak ada seed yang diterapkan.

4. Random Forest (RF). Algoritma _ensemble learning_ yang menggabungkan banyak decision trees untuk membuat prediksi. Setiap tree dilatih pada subset data yang berbeda, dan hasilnya digabungkan menggunakan voting mayoritas untuk klasifikasi. Random Forest sangat akurat dan dapat menangani dataset besar, namun bisa lebih lambat daripada model lain saat jumlah tree besar.

   Parameter default:

   - `n_estimators=100` Jumlah pohon (trees) yang dibangun dalam hutan.
   - `max_depth=None` Setiap pohon dapat tumbuh tanpa batasan kedalaman kecuali semua daun murni.
   - `random_state` Default-nya adalah None, artinya tidak ada seed yang diterapkan.

5. Gradient Boosting (GB). Algoritma ini membangun model secara bertahap, dengan fokus pada memperbaiki kesalahan model sebelumnya. Ini adalah pendekatan yang kuat namun lebih sensitif terhadap overfitting dibanding Random Forest. Gradient Boosting cocok untuk dataset yang kompleks namun memerlukan tuning parameter yang lebih hati-hati.

   Parameter default:

   - `n_estimators=100` Jumlah pohon yang dibangun secara bertahap, yaitu 100 pohon.
   - `learning_rate=0.1` Kecepatan pembelajaran yang mengontrol kontribusi masing-masing pohon ke model akhir. Yaitu laju pembelajaran sebesar 0.1.
   - `max_depth=3` Membatasi kedalaman maksimum setiap pohon 3, mencegah overfitting.
   - `random_state` Default-nya adalah None, artinya tidak ada seed yang diterapkan.

6. K-Nearest Neighbors (KNN). Algoritma ini membuat prediksi berdasarkan kedekatan antara sampel baru dengan sampel yang sudah dilabeli dalam data pelatihan. KNN mencari k tetangga terdekat dan memilih kelas mayoritas di antara tetangga tersebut. KNN bekerja dengan baik untuk dataset kecil namun lambat pada dataset besar.

   Parameter default:

   - `n_neighbors=5` Menggunakan 5 tetangga terdekat untuk prediksi.
   - `metric='minkowski'` Menggunakan metrik jarak Minkowski untuk menghitung jarak antara titik data.
   - `p=2` Nilai p=2 merepresentasikan jarak Euclidean.



## Evaluation
Setelah model dibuat, model-model tersebut selanjutnya dilakukan proses training. Proses training ini bertujuan untuk melatih model agar mampu mengenali data baru yang dimasukan oleh user. Berdasarkan prses trining yang sudah dilakukan Decision Tree, Random Forest, dan Gradient Boosting memberikan akurasi tertinggi, dimana seluruh data bisa dikenali dengan benar. Sementara itu, Logistic Regresion dan KNN memberikan akurasi yang sama yaitu 99.5%, sedangkan Naive Bayes meberikan hasil terendah yaitu 89.5%. Setelah proses training dilakukan, proses berikutnya adalah melakukan testing terhadap model yang dibuat. Proses ini bertujuan untuk mengetahui seberapa bagus model yang dibuat dalam mengenali data yang belum pernah dilihatnya. Berdasarkan hasil uji coba yang dilakukan, algoritma Naive Bayes meberikan akurasi sebesar 90%, sementara itu Logistic Regresion akurasi yang diperoleh sebesar 99%. Decision Tree, Random Forest, dan Gradient Boosting memberikan hasil maksimal, dimana akurasi yang diperoleh sebersa 100%. Sementara itu, algortima KNN memberikan hasil akurasi sebesar 99%.

Metrik Evaluasi

Metrik evaluasi yang digunakan dalam proyek ini adalah Accuracy, Precision, dan Recall. 
1. Accuracy: Mengukur persentase prediksi yang benar.
2. Precision: Mengukur ketepatan prediksi positif.
3. Recall: Mengukur kemampuan model dalam mendeteksi semua kasus positif.

## Kesimpulan
Berdasarkan hasil proses training dan testing pada berbagai algoritma, dapat disimpulkan bahwa Decision Tree, Random Forest, dan Gradient Boosting menunjukkan performa terbaik dengan akurasi 100% pada kedua proses. Logistic Regression dan KNN memberikan hasil yang hampir setara, dengan akurasi 99.5% pada training dan 99% pada testing. Sementara itu, Naive Bayes memberikan akurasi terendah, yaitu 89.5% pada training dan 90% pada testing. Hal ini menunjukkan bahwa meskipun Naive Bayes dapat digunakan, algoritma lain seperti Decision Tree, Random Forest, dan Gradient Boosting lebih efektif dalam mengenali data baik pada tahap pelatihan maupun pengujian.


```python

```
