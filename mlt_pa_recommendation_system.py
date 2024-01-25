# -*- coding: utf-8 -*-
"""MLT_PA_Recommendation_System.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/148RcxXkO_Vq2fmxQZ7Ut0XBWplR30Q0C

# Movie Recommendation System

## Data Understanding
"""

import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

!pip install kaggle

from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d nicoletacilibiu/movies-and-ratings-for-recommendation-system

# Ekstraksi File zip
local_zip = '/content/movies-and-ratings-for-recommendation-system.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content')
zip_ref.close()

# Membaca File
movies = pd.read_csv('/content/movies.csv')
ratings = pd.read_csv('/content/ratings.csv')

# Menampilkan Jumlah Data Unik
print('Jumlah data movies: ', len(movies['movieId'].unique()))
print('Jumlah data ratings: ', len(ratings['movieId'].unique()))

"""## Univariate Exploratory Data Analysis"""

# Informasi tentang Dataset Movies
movies.info()

# Informasi tentang Dataset Ratings
ratings.info()

# Menampilkan Data Movies
movies.head()

# Menampilkan Data Ratings
ratings.head()

# Deskripsi Statistik Data Ratings
ratings.describe()

# Menampilkan Jumlah Data
print('Jumlah userID: ', len(ratings.userId.unique()))
print('Jumlah movieId: ', len(ratings.movieId.unique()))
print('Jumlah data rating: ', len(ratings))

"""## Data Preparation"""

# Menambahkan Kolom `year_of_release` pada Data Movies
movies['year_of_release'] = movies.title.str.extract('([0-9]{4})')
movies.head()

# Mengonversi kolom 'title' menjadi tipe data string jika belum
movies['title'] = movies['title'].astype(str)

# Menggunakan metode split() untuk memisahkan judul film dari tahun rilis
movies['title'] = movies['title'].str.split(pat='(', n=1).str[0].str.strip()
movies.head()

# Periksa Nilai di kolom `rating`
ratings.rating.unique()

# Membulatkan Nilai Kolom 'rating'
ratings['rating'] = ratings['rating'].apply(np.ceil)

# Periksa Kembali Nilai di kolom `rating`
ratings.rating.unique()

# Mengubah Format Waktu pada Kolom 'timestamp' dari Data `ratings`
ratings.timestamp = pd.to_datetime(ratings['timestamp'], unit='s')
ratings.head()

# Menggabungkan Data Movies dan Ratings berdasarkan 'movieId'
df = pd.merge(movies, ratings, on='movieId', how='left')
df

# Mendeteksi `missing value` dengan fungsi isnull()
df.isnull().sum()

# Menghapus `missing value` dengan fungsi dropna()
df_clean = df.dropna()

# Cek Kembali `missing value`
df_clean.isnull().sum()

# Mengurutkan DataFrame berdasarkan Kolom 'movieId'
df_fix = df_clean.sort_values('movieId', ascending=True)
df_fix

# Menampilkan Jumlah Data Unik pada Kolom 'movieId'
len(df_fix.movieId.unique())

# Menampilkan Nilai Unik pada Kolom 'genres'
df_fix.genres.unique()

# Mengecek genre movies `(no genres listed)`
df_fix[df_fix['genres']=='(no genres listed)']

# Menghapus Baris dengan Genre `(no genres listed)`
df_fix = df_fix[(df_fix.genres != '(no genres listed)')]

# Membuat variabel preparation yang berisi dataframe `df_fix` kemudian mengurutkan berdasarkan `movieId`
preparation = df_fix
preparation.sort_values('movieId')

# Menghapus Duplikat Berdasarkan Kolom 'movieId' dan 'title' pada DataFrame Preparation
preparation = preparation.drop_duplicates('movieId')
preparation = preparation.drop_duplicates('title')
preparation

# Mengganti Nilai pada Kolom 'genres' dalam DataFrame Preparation
# Nilai 'Sci-Fi' diganti dengan 'Scifi' menggunakan regex
preparation = preparation.replace(to_replace ='[nS]ci-Fi', value = 'Scifi', regex = True)
preparation.head()

# Mengonversi data series `movieId` menjadi dalam bentuk list
movie_id = preparation['movieId'].tolist()

# Mengonversi data series `title` menjadi dalam bentuk list
movie_title = preparation['title'].tolist()

# Mengonversi data series `genres` menjadi dalam bentuk list
movie_genre = preparation['genres'].tolist()

print(len(movie_id))
print(len(movie_title))
print(len(movie_genre))

# Membuat dictionary untuk data `movie_id`, `movie_title`, dan `movie_genre`
df_movies = pd.DataFrame({
    'id': movie_id,
    'judul': movie_title,
    'genre': movie_genre
})
df_movies

"""## Model Development

### Content Based Filtering
"""

# Menampilkan Sampel Acak dari DataFrame 'df_movies'
data = df_movies
data.sample(5)

"""#### TF-IDF Vectorizer"""

# Inisialisasi dan Pemrosesan TfidfVectorizer pada Data Genre

# Menggunakan TfidfVectorizer untuk menghitung skor TF-IDF pada data genre.
tf = TfidfVectorizer()

# Melakukan perhitungan idf pada data genre.
tf.fit(data['genre'])

# Mengambil daftar fitur (feature names) yang digunakan dalam perhitungan TF-IDF.
tf.get_feature_names_out()

# Transformasi TF-IDF pada Data Genre dan Memeriksa Ukuran Matrix

# Menghitung skor TF-IDF dan mentransformasikan data genre ke dalam bentuk matrix.
tfidf_matrix = tf.fit_transform(data['genre'])

# Menampilkan ukuran matrix hasil transformasi TF-IDF.
tfidf_matrix.shape

# Mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense()
tfidf_matrix.todense()

# Membuat dataframe untuk melihat tf-idf matrix
# Kolom diisi dengan genre
# Baris diisi dengan judul

# Menampilkan sampel dari DataFrame
pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tf.get_feature_names_out(),
    index=data.judul
).sample(20, axis=1).sample(10, axis=0)

"""#### Cosine Similarity"""

# Menghitung cosine similarity pada matrix tf-idf
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim

# Membuat dataframe dari variabel cosine_sim dengan baris dan kolom berupa Judul Film
cosine_sim_df = pd.DataFrame(cosine_sim, index=data['judul'], columns=data['judul'])
print('Shape:', cosine_sim_df.shape)

# Melihat similarity matrix pada setiap Movie
cosine_sim_df.sample(5, axis=1).sample(10, axis=0)

"""#### Get Recommendations"""

def movie_recommendations(judul, similarity_data=cosine_sim_df, items=data[['judul', 'genre']], k=5):

    """
    Rekomendasi Movies berdasarkan kemiripan dataframe

    Parameter:
    ---
    judul : tipe data string (str)
                Nama Movies (index kemiripan dataframe)
    similarity_data : tipe data pd.DataFrame (object)
                      Kesamaan dataframe, simetrik, dengan Movies sebagai
                      indeks dan kolom
    items : tipe data pd.DataFrame (object)
            Mengandung kedua nama dan fitur lainnya yang digunakan untuk mendefinisikan kemiripan
    k : tipe data integer (int)
        Banyaknya jumlah rekomendasi yang diberikan
    ---

    Pada index ini, kita mengambil k dengan nilai similarity terbesar
    pada index matrix yang diberikan (i).

    """

    # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung
    # Dataframe diubah menjadi numpy
    # Range(start, stop, step)
    # Mengambil data index
    index = similarity_data.loc[:,judul].to_numpy().argpartition(
        range(-1, -k, -1))

    # Mengambil data dengan similarity terbesar dari index yang ada
    closest = similarity_data.columns[index[-1:-(k+2):-1]]

    # Drop judul agar nama movie yang dicari tidak muncul dalam daftar rekomendasi
    closest = closest.drop(judul, errors='ignore')

    return pd.DataFrame(closest).merge(items).head(k)

# Membuat Contoh Data
data.sample(3)

# Menampilkan Informasi Film "Judul" dari DataFrame 'data'
# Menggunakan fungsi eq() untuk mencari baris yang memiliki Judul dalam DataFrame 'data'
data[data.judul.eq("Bill & Ted's Bogus Journey")]

# Mendapatkan Rekomendasi Movies
movie_recommendations("Bill & Ted's Bogus Journey")

"""### Collaborative Filtering"""

# Membaca Dataset
df = preparation
df

"""#### Data Preparation"""

# Mengubah userId menjadi list tanpa nilai yang sama
user_ids = df['userId'].unique().tolist()
print('list userID: ', user_ids)

# Melakukan encoding userId
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
print('encoded userId : ', user_to_user_encoded)

# Melakukan proses encoding angka ke userId
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
print('encoded angka ke userId: ', user_encoded_to_user)

# Mengubah movieId menjadi list tanpa nilai yang sama
movie_ids = df['movieId'].unique().tolist()
print('list movieId: ', movie_ids)

# Melakukan proses encoding movieId
movie_to_movie_encoded = {x: i for i, x in enumerate(movie_ids)}
print('encoded movieId : ', movie_to_movie_encoded)

# Melakukan proses encoding angka ke movieId
movie_encoded_to_movie = {i: x for i, x in enumerate(movie_ids)}
print('encoded angka ke movieId: ', movie_encoded_to_movie)

# Mapping userID ke dataframe user
df['user'] = df['userId'].map(user_to_user_encoded)

# Mapping movieId ke dataframe movie
df['movie'] = df['movieId'].map(movie_to_movie_encoded)

# Mendapatkan jumlah user
num_users = len(user_to_user_encoded)
print(num_users)

# Mendapatkan jumlah movie
num_movie = len(movie_encoded_to_movie)
print(num_movie)

# Mengubah rating menjadi nilai float
df['rating'] = df['rating'].values.astype(np.float32)

# Nilai minimum rating
min_rating = min(df['rating'])

# Nilai maksimal rating
max_rating = max(df['rating'])

print('Number of User: {}, Number of Movie: {}, Min Rating: {}, Max Rating: {}'.format(
    num_users, num_movie, min_rating, max_rating
))

"""#### Membagi Data untuk Training dan Validasi"""

# Mengacak Baris DataFrame
df = df.sample(frac=1, random_state=42)
df

# Membuat variabel x untuk mencocokkan data user dan Movie menjadi satu value
x = df[['user', 'movie']].values

# Membuat variabel y untuk membuat rating dari hasil
y = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

# Membagi menjadi 80% data train dan 20% data validasi
train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

print(x, y)

"""## Proses Training"""

# Kelas `RecommenderNet`
class RecommenderNet(tf.keras.Model):

    # Insialisasi fungsi
    def __init__(self, num_users, num_movie, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movie = num_movie
        self.embedding_size = embedding_size

        # Embedding layer untuk mewakili vektor user
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        # Embedding layer untuk mewakili bias user
        self.user_bias = layers.Embedding(num_users, 1)

        # Embedding layer untuk mewakili vektor movie (film)
        self.movie_embedding = layers.Embedding(
            num_movie,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        # Embedding layer untuk mewakili bias movie (film)
        self.movie_bias = layers.Embedding(num_movie, 1)

    def call(self, inputs):
        # Mendapatkan vektor user dan bias user
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])

        # Mendapatkan vektor movie (film) dan bias movie (film)
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])

        # Melakukan operasi dot product antara vektor user dan movie
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)

        # Menghitung prediksi akhir dengan menambahkan vektor dan bias user serta movie (film)
        x = dot_user_movie + user_bias + movie_bias

        # Menggunakan fungsi aktivasi sigmoid untuk mendapatkan nilai antara 0 dan 1
        return tf.nn.sigmoid(x)

model = RecommenderNet(num_users, num_movie, 50) # inisialisasi model

# model compile
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

# Membuat callback EarlyStopping
early_stopping_callback = EarlyStopping(
    patience=5,  # Menentukan berapa epoch tanpa peningkatan yang diizinkan
    restore_best_weights=True,  # Mengembalikan bobot terbaik jika pelatihan dihentikan
    monitor='val_root_mean_squared_error',  # Menentukan metrik yang akan dimonitor
    mode='min'  # Mode 'min' berarti pelatihan dihentikan saat metrik berhenti menurun
)

# Memulai training dengan callback
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=32,
    epochs=100,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping_callback]
)

"""## Evaluation"""

# Evaluasi Model pada Data Uji
test_loss, test_rmse = model.evaluate(x, y)
print(f'Test Loss: {test_loss}, Test RMSE: {test_rmse}')

# Visualisasi Metrik Evaluasi

plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('Model Metrics')
plt.ylabel('RMSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Menyimapan Model Rekomendasi
model.save('my_recommendation_model', save_format='tf')

# Memuat model dari format SavedModel
loaded_model = tf.keras.models.load_model('my_recommendation_model')

"""## Get Movie Recommendations"""

# Membaca Data
movie_df = preparation

# Mengambil sample user
user_id = df.userId.sample(1).iloc[0]
movie_visited_by_user = df[df.userId == user_id]

# Operator bitwise (~), bisa diketahui di sini https://docs.python.org/3/reference/expressions.html
movie_not_visited = movie_df[~movie_df['movieId'].isin(movie_visited_by_user.movieId.values)]['movieId']
movie_not_visited = list(
    set(movie_not_visited)
    .intersection(set(movie_to_movie_encoded.keys()))
)

movie_not_visited = [[movie_to_movie_encoded.get(x)] for x in movie_not_visited]
user_encoder = user_to_user_encoded.get(user_id)
user_movie_array = np.hstack(
    ([[user_encoder]] * len(movie_not_visited), movie_not_visited)
)

# Mengambil sample user
user_id = movie_df.userId.sample(1).iloc[0]
movie_visited_by_user = movie_df[movie_df.userId == user_id]

# Operator bitwise (~), bisa diketahui di sini https://docs.python.org/3/reference/expressions.html
movie_not_visited = movie_df[~movie_df['movieId'].isin(movie_visited_by_user.movieId.values)]['movieId']
movie_not_visited = list(
    set(movie_not_visited)
    .intersection(set(movie_to_movie_encoded.keys()))
)

movie_not_visited = [[movie_to_movie_encoded.get(x)] for x in movie_not_visited]
user_encoder = user_to_user_encoded.get(user_id)
user_movie_array = np.hstack(
    ([[user_encoder]] * len(movie_not_visited), movie_not_visited)
)

ratings = model.predict(user_movie_array).flatten()

top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_movie_ids = [
    movie_encoded_to_movie.get(movie_not_visited[x][0]) for x in top_ratings_indices
]

print('Menampilkan Rekomendasi untuk Pengguna (User): {}'.format(user_id))
print('===' * 9)
print('Film dengan Rating Tinggi dari Pengguna (User)')
print('----' * 8)

top_movie_user = (
    movie_visited_by_user.sort_values(
        by='rating',
        ascending=False
    )
    .head(5)
    .movieId.values
)

movie_df_rows = movie_df[movie_df['movieId'].isin(top_movie_user)]
for row in movie_df_rows.itertuples():
    print(row.title, ':', row.genres)

print('----' * 8)
print('10 Rekomendasi Film Teratas')
print('----' * 8)

recommended_movie = movie_df[movie_df['movieId'].isin(recommended_movie_ids)]
for row in recommended_movie.itertuples():
    print(row.title, ':', row.genres)