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
- Memanfaatkan metode content-based filtering untuk merekomendasikan film dengan karakteristik serupa yang sesuai dengan selera pengguna.
#### Implementasi _Collaborative Filtering_
- Mengintegrasikan _collaborative filtering_ untuk memanfaatkan pola perilaku pengguna yang mirip.
- Menyusun model rekomendasi berdasarkan data pengguna lain dengan preferensi serupa, sehingga meningkatkan akurasi rekomendasi.

## Data Understanding
Dalam Proyek ini, dataset yang digunakan untuk pengembangan sistem rekomendasi ini diperoleh dari [Kaggle Datasets](https://www.kaggle.com/datasets). Dataset ini berbentuk file zip yang terdiri dari dua file CSV terpisah, yaitu `movies.csv` yang memiliki total 9742 _rows_ × 3 _columns_ dengan 1 fitur bertipe data _int64_ & 2 fitur bertipe data _object_, dan `ratings.csv` yang memiliki total 100854 _rows_ × 7 _columns_ dengan 3 fitur bertipe data _int64_ & 1 fitur bertipe data _float64_.

## Data Preparation


## Modeling


## Evaluation

