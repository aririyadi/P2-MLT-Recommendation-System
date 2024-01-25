# Laporan Proyek Machine Learning - _Ari Riyadi_

## Project Overview

Pesatnya pertumbuhan pengumpulan data telah membawa era informasi baru. Data digunakan untuk menciptakan sistem yang lebih efisien dan di sinilah Sistem Rekomendasi berperan. Sistem Rekomendasi adalah jenis sistem penyaringan informasi yang meningkatkan kualitas hasil pencarian dan menyediakan item yang lebih relevan dengan item pencarian atau berhubungan dengan riwayat pencarian pengguna.

Proyek ini bertujuan untuk mengembangkan sistem rekomendasi film menggunakan teknik machine learning. Saat ini, di era digital, masyarakat memiliki akses yang melimpah ke berbagai konten audio-visual, termasuk film. Namun, jumlah film yang tersedia dapat membuat pengguna bingung untuk memilih film yang sesuai dengan preferensi mereka. Sistem rekomendasi film adalah solusi yang efektif untuk memberikan rekomendasi yang personal dan relevan kepada pengguna berdasarkan sejarah penontonannya.

Proyek ini penting karena memberikan solusi bagi masalah kelebihan informasi di dunia digital. Dengan pertumbuhan platform streaming dan jumlah film yang terus meningkat, pengguna sering kesulitan menemukan film yang sesuai dengan minat mereka. Sistem rekomendasi dapat meningkatkan pengalaman pengguna, membantu mereka menemukan konten yang lebih relevan, dan pada gilirannya, meningkatkan kepuasan pelanggan.

### References and Related Research

- [Introduction to Recommender System](https://hackernoon.com/introduction-to-recommender-system-part-1-collaborative-filtering-singular-value-decomposition-44c9659c5e75)
- [Getting Started with a Movie Recommendation System](https://www.kaggle.com/code/ibtesama/getting-started-with-a-movie-recommendation-system)
- [Movie Recommendation system(For Deployment)](https://www.kaggle.com/code/terminate9298/movie-recommendation-system-for-deployment)
- [Sistem Rekomendasi Film Menggunakan Content Based Filtering](https://j-ptiik.ub.ac.id/index.php/j-ptiik/article/download/9163/4159/)

## Business Understanding

### Problem Statements
- Banyaknya pilihan film di platform streaming membuat pengguna kesulitan dalam menemukan konten yang sesuai dengan preferensi mereka.
- Keterbatasan sistem rekomendasi saat ini untuk memberikan rekomendasi yang akurat dan personal dapat mengakibatkan pengguna kehilangan minat dalam mengeksplorasi konten baru.

### Goals
- Mengembangkan sistem rekomendasi yang dapat memberikan rekomendasi film yang sesuai dengan preferensi pengguna, mengatasi masalah kelebihan informasi, dan meningkatkan pengalaman pengguna.
- Meningkatkan akurasi rekomendasi dengan memanfaatkan teknik-teknik machine learning seperti _content-based filtering_ dan _collaborative filtering_, sehingga pengguna merasa lebih terhubung dengan konten yang mereka nikmati.

### Solution statements
#### Pengembangan Algoritma _Content-Based Filtering_
- Membangun algoritma yang dapat menganalisis preferensi pengguna berdasarkan sejarah penontonannya, penilaian atau ulasan sebelumnya.
- Memanfaatkan metode _content-based filtering_ untuk merekomendasikan film dengan karakteristik serupa yang sesuai dengan selera pengguna.
#### Implementasi _Collaborative Filtering_
- Mengintegrasikan _collaborative filtering_ untuk memanfaatkan pola perilaku pengguna yang mirip.
- Menyusun model rekomendasi berdasarkan data pengguna lain dengan preferensi serupa, sehingga meningkatkan akurasi rekomendasi.

## Data Understanding
Dalam Proyek ini, dataset yang digunakan untuk pengembangan sistem rekomendasi ini diperoleh dari [Kaggle Datasets](https://www.kaggle.com/datasets). Dataset ini berbentuk file zip yang terdiri dari dua file CSV terpisah, yaitu file `movies.csv` yang memiliki total 9742 _rows_ × 3 _columns_ dengan 1 fitur bertipe data _int64_ & 2 fitur bertipe data _object_, dan file `ratings.csv` yang memiliki total 100854 _rows_ × 4 _columns_ dengan 3 fitur bertipe data _int64_ & 1 fitur bertipe data _float64_.
Dataset ini dapat diunduh melalui situs [Kaggle : Movies & Ratings for Recommendation System](https://www.kaggle.com/datasets/nicoletacilibiu/movies-and-ratings-for-recommendation-system).

### Variabel-Variabel Pada Dataset:
**Fitur movies atau dataset `movies.csv`:** Merupakan kumpulan data yang berisi informasi mengenai film-film. Setiap baris dalam dataset ini merepresentasikan satu film, dan kolom-kolomnya memberikan detail tentang atribut-atribut film tersebut.

- `movieId`: Informasi nilai unik untuk setiap film.
- `title`: Informasi judul untuk setiap film.
- `genres`: Informasi mengenai genre-genre yang terkandung dalam setiap film.

**Fitur ratings atau dataset `ratings.csv`:** Merupakan kumpulan data yang berisi informasi tentang penilaian yang diberikan oleh pengguna terhadap film-film dalam dataset. Setiap baris dalam dataset ini merepresentasikan satu penilaian dari seorang pengguna terhadap suatu film.

- `userId`: Informasi nilai unik untuk setiap pengguna.
- `movieId`: Informasi nilai unik untuk setiap film.
- `rating`: Informasi penilaian numerik yang diberikan oleh pengguna terhadap film tertentu. 
- `timestamp`: Informasi waktu ketika penilaian diberikan oleh pengguna.

### Univariate Exploratory Data Analysis
#### Eksplorasi Variabel
- Eksplorasi variabel menggunakan fungsi `movies.info()` pada variabel _movies_.
- Eksplorasi variabel menggunakan fungsi `ratings.info()` pada variabel _ratings_.
- Menampilkan fitur _movies_ menggunakan fungsi `movies.head()`.

**Tabel 1**. Informasi pada fitur _movies_
| movieId | title                                  | genres                                          |
|---------|----------------------------------------|-------------------------------------------------|
| 1       | Toy Story (1995)                       | Adventure|Animation|Children|Comedy|Fantasy     |
| 2       | Jumanji (1995)                         | Adventure|Children|Fantasy                      |
| 3       | Grumpier Old Men (1995)                | Comedy|Romance                                  |
| 4       | Waiting to Exhale (1995)               | Comedy|Drama|Romance                            |
| 5       | Father of the Bride Part II (1995)     | Comedy                                          |

- Menampilkan fitur _ratings_ menggunakan fungsi `ratings.head()`.

**Tabel 2**. Informasi pada fitur _movies_
| userId | movieId | rating | timestamp  |
|--------|---------|--------|------------|
| 1      | 1       | 4.0    | 964982703  |
| 1      | 3       | 4.0    | 964981247  |
| 1      | 6       | 4.0    | 964982224  |
| 1      | 47      | 5.0    | 964983815  |
| 1      | 50      | 5.0    | 964982931  |

- mendeskripsikan fitur _ratings_ menggunakan fungsi `ratings.describe()`.

**Tabel 3**. Informasi statistik pada fitur ratings
|           | userId           | movieId          | rating           | timestamp          |
|-----------|------------------|------------------|------------------|--------------------|
| **count** | 100836.000000    | 100836.000000    | 100836.000000    | 1.008360e+05       |
| **mean**  | 326.127564       | 19435.295718     | 3.501557         | 1.205946e+09       |
| **std**   | 182.618491       | 35530.987199     | 1.042529         | 2.162610e+08       |
| **min**   | 1.000000         | 1.000000         | 0.500000         | 8.281246e+08       |
| **25%**   | 177.000000       | 1199.000000      | 3.000000         | 1.019124e+09       |
| **50%**   | 325.000000       | 2991.000000      | 3.500000         | 1.186087e+09       |
| **75%**   | 477.000000       | 8122.000000      | 4.000000         | 1.435994e+09       |
| **max**   | 610.000000       | 193609.000000    | 5.000000         | 1.537799e+09       |

**Keterangan:**

- **Count** adalah jumlah sampel pada data.
- **Mean** adalah nilai rata-rata.
- **Std** adalah standar deviasi.
- **Min** adalah nilai minimum setiap kolom. 
- **25%** adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama. 
- **50%** adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
- **75%** adalah kuartil ketiga.
- **Max** adalah nilai maksimum.

## Data Preparation

#### Memisahkan Judul Film dari Tahun Rilis:
- Menggunakan metode `split()` untuk mendapatkan judul film tanpa tahun rilis.
- Alasan: Untuk memudahkan analisis data.

#### Membulatkan Nilai pada Kolom 'rating':
- Menggunakan fungsi `np.ceil` untuk membulatkan nilai rating 
- Alasan: Untuk menyesuaikan skala penilaian dengan nilai bulat.

#### Mengubah Format Waktu pada Kolom 'timestamp':
- Menggunakan fungsi `pd.to_datetime` untuk mengonversi timestamp ke format datetime.
- Alasan: Memudahkan analisis berbasis waktu.

#### Menggabungkan Data _Movies_ dan _Ratings_ berdasarkan 'movieId':
- Menggunakan metode `merge` untuk menggabungkan data _movies_ dan _ratings_ berdasarkan 'movieId'.
- Alasan: Menggabungkan informasi film dan rating pengguna.

#### Mendeteksi dan Menghapus _Missing Values_:
- Menggunakan `isnull()` untuk mendeteksi _missing values_.
- Menggunakan `dropna()` untuk menghapus baris dengan _missing values_.
- Alasan: Membersihkan data dari nilai yang hilang.

#### Menghapus Duplikat Berdasarkan Kolom 'movieId' dan 'title':
- Menggunakan `drop_duplicates` untuk menghapus duplikat berdasarkan 'movieId' dan 'title'.
- Alasan: Memastikan setiap film hanya muncul sekali dalam dataset.

#### Mengonversi Data _Series_ menjadi _List_:
- Mengonversi 'movieId', 'title', dan 'genres' menjadi list.
- Alasan: Persiapan untuk pembuatan dictionary dan analisis lebih lanjut.

#### Membuat _Dictionary_ dari Data Series:
- Membuat _dictionary_ dari data 'movieId', 'title', dan 'genres'.
- Alasan: Mempermudah akses dan penggunaan data.

## Modeling
Pada proyek ini, implementasi 2 pendekatan dengan menggunakan teknik _content-based filtering_ dan _collaborative filtering_.

### _Content-Based Filtering_
_Content based filtering_ pada Sistem rekomendasi adalah metode yang mempertimbangkan perilaku dari pengguna dari masa lalu yang kemudian diidentifikasi pola perilakunya untuk merekomendasikan barang atau sesuatu yang sesuai dengan pola perilaku tersebut. Metode _content based filtering_ menganalisis preferensi dari perilaku pengguna dimasa lalu untuk membuat model. Model tersebut akan dicocokkan dengan serangkaian karakteristik atribut dari barang yang akan direkomendasikan. Barang dengan tingkat kecocokan tertinggi akan menjadi rekomendasi untuk pengguna. Contoh sederhana, jika seseorang menyukai suatu barang tertentu, dia juga akan menyukai barang yang serupa dengannya. 

Berikut proses atau tahapan dalam implementasi _content-based filtering_ menggunakan _TF-IDF Vectorizer_ dan _Cosine Similarity_:
- Menyiapkan _dataframe_ yang telah dibersihkan pada tahap data _preparation_ sebelumnya.
- Menggunakan _TfidfVectorizer_ untuk menghitung skor 'TF-IDF' pada data genre.
- Menghitung _Cosine Similarity_ pada hasil 'TF-IDF' untuk mendapatkan _similarity matrix_.
- Melakukan Transformasi 'TF-IDF' pada Data Genre dan Memeriksa Ukuran _Matrix_.
- Membuat fungsi _movie_recommendations_ untuk mendapatkan rekomendasi berdasarkan _similarity matrix_.

#### Kelebihan _Content-Based Filtering_:
- Menyediakan rekomendasi yang lebih personal karena mempertimbangkan preferensi individu pengguna berdasarkan sejarah interaksi mereka.
- Lebih baik dalam menangani masalah _cold start_, yaitu memberikan rekomendasi untuk item baru atau pengguna baru karena tidak sepenuhnya bergantung pada perilaku pengguna sebelumnya.
- Lebih transparan karena dasar rekomendasi dapat dijelaskan dengan mudah, terutama jika diperhitungkan atribut atau karakteristik spesifik item yang digunakan dalam proses rekomendasi.
- Tidak memerlukan data eksternal tentang perilaku pengguna lainnya, dan dapat berfungsi cukup baik bahkan ketika informasi kolaboratif tentang pengguna tidak tersedia.

#### Kekurangan _Content-Based Filtering_:
- Cenderung menghasilkan rekomendasi yang terbatas pada jenis item yang sudah diketahui pengguna. Dengan kata lain, mungkin tidak efektif dalam menemukan item baru atau eksplorasi.
- Kinerjanya tergantung pada seberapa baik konten item direpresentasikan dan dianalisis. Jika representasi konten tidak akurat atau lengkap, maka kualitas rekomendasi dapat menurun.
- Ada risiko _over-specialization_, di mana rekomendasi menjadi terlalu fokus pada preferensi yang spesifik, mengabaikan variasi atau kejutan yang mungkin diinginkan pengguna.
- Sulit menangani perubahan drastis dalam preferensi pengguna karena cenderung mempertahankan model yang ada dan kurang fleksibel dalam menyesuaikan perubahan tersebut.
- Bergantung pada informasi yang ada dalam deskripsi konten item, sehingga tidak efektif untuk item yang kurang memiliki deskripsi atau informasi konten yang relevan.

### _Collaborative Filtering_
_Collaborative filtering_ adalah suatu teknik yang umum digunakan dalam sistem rekomendasi untuk memberikan saran yang dipersonalisasi kepada pengguna. Ide dasar di balik _collaborative filtering_ adalah membuat prediksi tentang preferensi atau minat pengguna dengan mengumpulkan preferensi atau perilaku dari beberapa pengguna (kolaborator). pada tahap ini, _Collaborative filtering_ menggunakan pendekatan _embedding_ dengan model _RecommenderNet_. Berikut proses atau tahapan dalam implementasi _collaborative filtering_:

#### 1. Data Preparation
Melakukan _encode_ pada feature 'userId' dan 'movieId'. proses _encode_ akan memetakan setiap nilai pada kedua _feature_ tersebut ke dalam bentuk _index_.
#### 2. Pembagian Data Train dan Validasi
Pembagian data train dan validasi dilakukan dengan komposisi 80:20. Pembagian ini bertujuan untuk mencapai sejumlah tujuan kritis dalam proses pengembangan model dan membantu memastikan kehandalan serta kinerja yang optimal pada berbagai kondisi penggunaan.
#### 3. Training
Proses training dilakukan dengan mengimplementasikan teknik _embedding_ pada model _RecommenderNet_ untuk menghitung skor kecocokan antara film dan pengguna. Dalam proses _compile_, _BinaryCrossentropy_ digunakan sebagai fungsi kerugian untuk tugas prediksi _biner_, sementara _Adam optimizer_ dengan _learning rate_ 0.001 dan _Root Mean Squared Error (RMSE)_ sebagai metrik evaluasi. Pelatihan model berlangsung selama 100 epochs dengan batch size 32, menggunakan data latih untuk melatih model dan data validasi untuk evaluasi. _Callback EarlyStopping_ diterapkan dengan _patience_ 5, sehingga pelatihan akan berhenti jika tidak ada peningkatan yang signifikan dalam metrik validasi selama 5 _epoch_ berturut-turut. Setelah pelatihan, model dievaluasi menggunakan data validasi untuk memastikan generalisasi yang baik.
#### 4. Metrik Evaluasi
![1](https://github.com/aririyadi/P2-MLT-Recommendation-System/assets/147322531/d2e1a3d6-c4fe-4dd1-af6f-65d58b394f70)

**Gambar 1**. Visualisasi Metrik Evaluasi

Perhatikan gambar diatas, proses training model menunjukkan bahwa model memiliki performa yang baik dan model konvergen pada _epochs_ sekitar 100. Dari proses ini, kita memperoleh nilai error akhir sebesar sekitar 0.0710 dan error pada data validasi sebesar 0.2618. Nilai tersebut dapat digunakan untuk membuat sistem rekomendasi.

#### 5. Mendapatkan Rekomendasi
Untuk mendapatkan rekomendasi _movie_ atau film, pertama kita ambil sampel _user_ secara acak dan definisikan variabel `movie_not_visited` yang merupakan daftar _movie_ yang belum pernah dikunjungi oleh pengguna. Selanjutnya, untuk memperoleh rekomendasi _movie_ atau film, gunakan fungsi `model.predict()` dari library Keras, kemudian sistem akan memberikan rekomendasi sebagai berikut:
```sh
Menampilkan Rekomendasi untuk Pengguna (User): 599.0
===========================
Film dengan Rating Tinggi dari Pengguna (User)
--------------------------------
His Girl Friday : Comedy|Romance
Key Largo : Crime|Drama|Film-Noir|Thriller
On the Beach : Drama
Badlands : Crime|Drama|Thriller
Mulholland Dr. : Drama|Mystery|Romance
--------------------------------
10 Rekomendasi Film Teratas
--------------------------------
Surviving the Game : Action|Adventure|Thriller
Jaws : Action|Horror
Shooting Fish : Comedy|Romance
Woman in the Dunes : Drama
Cup, The : Comedy
Frequency : Drama|Thriller
In the Mood For Love : Drama|Romance
Widow of St. Pierre, The : Drama|Romance
Painted Veil, The : Drama|Romance
Raiders of the Lost Ark: The Adaptation : Action|Adventure|Thriller
```
#### Kelebihan _Collaborative Filtering_:
- _Collaborative Filtering_ memberikan rekomendasi yang personal karena didasarkan pada preferensi dan perilaku pengguna sejenis.
- Tidak memerlukan informasi eksplisit tentang item, sehingga cocok untuk sistem dengan banyak item atau di mana deskripsi item sulit didapatkan.
- Dapat menangani produk baru atau item yang belum pernah dihadapi sebelumnya karena rekomendasi didasarkan pada perilaku pengguna.
- Secara alami dapat menangkap tren dan perubahan dalam preferensi pengguna seiring waktu.
- Efektif pada dataset dengan dimensi rendah, di mana tidak ada banyak atribut atau metadata yang terkait dengan item.

#### Kekurangan _Collaborative Filtering_:
- Mengalami kesulitan dalam memberikan rekomendasi untuk pengguna atau item baru yang belum memiliki sejarah preferensi atau interaksi.
- Rentan terhadap masalah sparsitas pada data, terutama jika sebagian besar pengguna hanya berinteraksi dengan sebagian kecil item.
- Skalabilitas dapat menjadi masalah ketika jumlah pengguna dan item sangat besar, karena perhitungan kesamaan antar pengguna atau item dapat menjadi rumit.
- Model hanya dapat merekomendasikan item berdasarkan kesamaan dengan pengguna lain yang serupa. Ini dapat menghasilkan rekomendasi yang kurang beragam.
- Bergantung pada perilaku pengguna yang sudah ada, sehingga dapat memberikan rekomendasi yang kurang akurat jika aktivitas pengguna tiba-tiba berubah.
- Kesulitan dalam menangani item atau konten baru yang belum memiliki cukup data untuk dibandingkan dengan pengguna lain.

## Evaluation

### _Content-Based Filtering_
Pada pendekatan _Content-Based Filtering_, performa model diukur menggunakan nilai metrik _precision_ dengan _similarity_. _Cosinus Similarity_ digunakan sebagai ukuran yang mengkuantifikasi kesamaan antara vektor. _Precision_ merupakan tingkat ketepatan antara informasi yang diminta pengguna dengan hasil yang diberikan oleh sistem.

#### Formula _Precision_:

$\ \text{Precision} = \frac{\text{True Positive (TP)}}{\text{True Positive (TP) + False Positive (FP)}} \$

**Keterangan:**
- **True Positive (TP):** Prediksi yang benar positif, di mana model memprediksi dengan tepat bahwa suatu item relevan.
- **False Positive (FP):** Prediksi yang salah positif, di mana model memprediksi bahwa suatu item relevan, padahal sebenarnya tidak.

Formula _precision_ ini memberikan gambaran tentang seberapa banyak dari item yang direkomendasikan oleh model yang benar-benar relevan dengan preferensi pengguna. _Precision_ dapat diinterpretasikan sebagai "dari item yang direkomendasikan, berapa persen yang benar-benar disukai oleh pengguna." Dalam proyek ini, _precision_ dihitung berdasarkan prediksi relevan terhadap item yang direkomendasikan.

#### Formula _Cosinus Similarity_:

$\ \text{Cosinus Similarity} = \frac{\sum_{i=1}^{n} A_i \times B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \times \sqrt{\sum_{i=1}^{n} B_i^2}} \$

**Keterangan:**
- $\( A_i \)$ dan $\( B_i \)$ adalah komponen vektor dari dua item yang dibandingkan.
- $\( n \)$ adalah jumlah fitur atau dimensi dalam vektor.

_Cosinus Similarity_ mengukur kesamaan antara dua vektor berdasarkan _cosinus_ dari sudut antara vektor-vektor tersebut. Semakin besar nilai _Cosinus Similarity_, semakin mirip kedua vektor tersebut. Dalam konteks _Content-Based Filtering_, vektor ini mewakili representasi fitur dari item yang dibandingkan.

#### Evaluasi Menggunakan Pendekatan _Content-Based Filtering_
```sh
data[data.judul.eq("Bill & Ted's Bogus Journey")]
```
Output:

**Tabel 4**. Hasil Pencarian Film "Bill & Ted's Bogus Journey"
| id   | judul                        | genre                           |
|------|------------------------------|---------------------------------|
| 4980 | Bill & Ted's Bogus Journey   | Adventure|Comedy|Fantasy|Scifi  |
```sh
movie_recommendations("Bill & Ted's Bogus Journey")
```
Output:

**Tabel 5**. Hasil Rekomendasi Film Berdasarkan "Bill & Ted's Bogus Journey"
| judul                        | genre                                  |
|------------------------------|----------------------------------------|
| Time Bandits                 | Adventure|Comedy|Fantasy|Scifi         |
| Mothra	                     | Adventure|Fantasy|Scifi                |
| Biggles                      | Adventure|Fantasy|Scifi                |
| Tin Man	                     | Adventure|Fantasy|Scifi                |
| Ant-Man and the Wasp         | Action|Adventure|Comedy|Fantasy|Scifi  |

- Film "Bill & Ted's Bogus Journey" memiliki genre Adventure, Comedy, Fantasy, dan Scifi.
- Model _Content-Based Filtering_ memberikan 5 rekomendasi film dengan genre yang mirip, termasuk Adventure, Comedy, Fantasy, dan Scifi.
- Dari 5 rekomendasi, semuanya memiliki genre yang sesuai dengan film yang dicari, menunjukkan konsistensi model dalam memberikan rekomendasi berdasarkan preferensi pengguna.

Jadi, _Cosinus Similarity_ membantu dalam mengukur kesamaan fitur antara film yang dicari dan film-film rekomendasi, sementara _Precision_ memberikan gambaran tentang seberapa baik model dapat memberikan rekomendasi yang sesuai dengan preferensi pengguna berdasarkan informasi fitur tersebut. Keduanya bekerja bersama untuk memberikan pemahaman yang komprehensif tentang kualitas rekomendasi yang diberikan oleh model _Content-Based Filtering_.

### _Collaborative Filtering_
Pada pendekatan _Collaborative Filtering_, metrik evaluasi yang digunakan adalah _Root Mean Squared Error (RMSE)_. _RMSE_ digunakan sebagai indikator seberapa baik model _Collaborative Filtering_ mampu memprediksi preferensi pengguna terhadap item dalam sistem rekomendasi. _RMSE_ mengukur deviasi rata-rata antara nilai sebenarnya dan nilai prediksi, memberikan gambaran tentang tingkat akurasi model.

#### Formula _RMSE_:

_RMSE_ dihitung dengan rumus berikut:

$\[ RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2} \]$

Di mana:
- $\( N \)$ adalah jumlah sampel atau pasangan pengguna-item.
- $\( y_i \)$ adalah nilai sebenarnya dari preferensi pengguna terhadap item.
- $\( \hat{y}_i \)$ adalah nilai prediksi model untuk preferensi pengguna terhadap item.

_RMSE_ mengukur deviasi rata-rata antara nilai sebenarnya dan nilai prediksi, memberikan gambaran tentang tingkat akurasi model. Semakin kecil nilai _RMSE_, semakin baik model dapat memprediksi preferensi pengguna terhadap item.

Berdasarkan hasil evaluasi, model _Collaborative Filtering_ menunjukkan kinerja yang baik dengan kemampuan untuk memprediksi preferensi pengguna secara akurat. Meskipun terdapat sedikit perbedaan antara hasil pada data latih dan data validasi, namun nilai _RMSE_ yang relatif kecil pada keduanya menandakan bahwa model tersebut dapat umumnya digeneralisasikan dengan baik pada data yang tidak terlihat selama pelatihan. Nilai _Root Mean Squared Error (RMSE)_ yang dicapai pada data latih sebesar 0.0710 dan pada data validasi sebesar 0.2618 menunjukkan bahwa model memiliki tingkat akurasi yang tinggi dalam memprediksi preferensi pengguna terhadap item. Semakin kecil nilai _RMSE_, semakin baik kemampuan model dalam membuat prediksi yang mendekati nilai sebenarnya. Dengan nilai _RMSE_ yang relatif kecil, model _Collaborative Filtering_ dapat dianggap berhasil dalam tugas rekomendasi item berdasarkan preferensi pengguna.

## Conclusion
Proses pengembangan sistem rekomendasi film menggunakan teknik machine learning dengan fokus pada _Content-Based Filtering_ dan _Collaborative Filtering_. Proyek ini mengintegrasikan algoritma _Content-Based Filtering_ yang menganalisis preferensi pengguna berdasarkan sejarah penontonan, penilaian, atau ulasan sebelumnya, serta _Collaborative Filtering_ yang memanfaatkan pola perilaku pengguna sejenis. Data yang digunakan berasal dari dataset Kaggle yang terdiri dari informasi film dan penilaian pengguna. _Content-Based Filtering_ menggunakan _TF-IDF Vectorizer_ dan _Cosine Similarity_ untuk memberikan rekomendasi berdasarkan kesamaan fitur genre. Sementara itu, _Collaborative Filtering_ menerapkan _embedding_ dengan model _RecommenderNet_ dan mengukur performanya menggunakan _Root Mean Squared Error (RMSE)_. Evaluasi kedua pendekatan menunjukkan bahwa keduanya memberikan rekomendasi yang sesuai dengan preferensi pengguna dengan akurasi yang tinggi, namun dengan kelebihan dan kekurangan masing-masing. _Content-Based Filtering_ menyediakan rekomendasi yang lebih personal dan dapat menangani _cold start_, tetapi terbatas pada jenis item yang sudah diketahui pengguna. Sementara _Collaborative Filtering_ memberikan rekomendasi yang beragam dan dapat menangani item baru, tetapi rentan terhadap _sparsitas_ data dan kesulitan dalam menangani perubahan drastis dalam preferensi pengguna. Keseluruhan, proyek ini memberikan pemahaman mendalam tentang implementasi kedua pendekatan dalam konteks sistem rekomendasi film.

## Reference
[1]. Ricci, F., Rokach, L., & Shapira, B. (2015). *Introduction to Recommender Systems Handbook*. Springer.

[2]. Adomavicius, G., & Tuzhilin, A. (2005). *Toward the Next Generation of Recommender Systems: A Survey of the State-of-the-Art and Possible Extensions*. IEEE Transactions on Knowledge and Data Engineering, 17(6), 734-749.

[3]. Desrosiers, C., & Karypis, G. (2011). *A Comprehensive Survey of Neighborhood-Based Recommendation Methods*. In *Recommender Systems Handbook* (pp. 107-144). Springer.

[4]. Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix Factorization Techniques for Recommender Systems*. Computer, 42(8), 30-37.
