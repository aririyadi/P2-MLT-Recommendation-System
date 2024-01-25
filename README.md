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
- **Min** yaitu nilai minimum setiap kolom. 
- **25%** adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama. 
- **50%** adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
- **75%** adalah kuartil ketiga.
- **Max** adalah nilai maksimum.

## Data Preparation


## Modeling


## Evaluation

