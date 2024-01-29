import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model

model = load_model('My_Model')

movies = pd.read_csv('movies.csv')

st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<h1 style='text-align: center;'>Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown(
    "<h6 style='text-align: center; color: grey;'>Oleh : <a href='https://aririyadi.github.io/myportfolio/' target='_blank'> Ari Riyadi </a></h6>",
    unsafe_allow_html=True)

"\n\n"

selected = option_menu(
    None,
    ['Content Based Filtering', 'Collaborative Filtering'],
    icons=['layers-fill', 'server'],
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0px!important", "background-color": "#ADD8E6", "font-family": "arial",
                      "letter-spacing": "0.5px"},
        "icon": {"color": "orange", "font-size": "18px"},
        "nav-link": {"font-size": "18px", "text-align": "center", "padding": "10px", "margin": "0px",
                     "--hover-color": "#eee", "color": "#2b2b2b"},
        "nav-link-selected": {"background-color": "#20B2AA"},
    }
)

if selected == 'Content Based Filtering':
    df_movies = pd.DataFrame({
        'Id': movies['movieId'].tolist(),
        'Judul Film': movies['title'].tolist(),
        'Genre': movies['genre'].tolist()
    })

    count_vectorizer = CountVectorizer()
    genre_matrix = count_vectorizer.fit_transform(df_movies['Genre'])
    feature_names = count_vectorizer.get_feature_names_out()

    count_matrix_df = pd.DataFrame(
        genre_matrix.toarray(),
        columns=feature_names,
        index=df_movies['Judul Film']
    )

    similarity_matrix = cosine_similarity(genre_matrix, genre_matrix)

    def recommend_movies(movie_title, similarity_matrix, df_movies, num_recommendations=10):
        df_movies['Judul Film'] = df_movies['Judul Film'].str.strip()
        movie_index = df_movies[df_movies['Judul Film'] == movie_title].index[0]
        sim_scores = list(enumerate(similarity_matrix[movie_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations + 1]
        movie_indices = [i[0] for i in sim_scores]

        recommended_movies = df_movies[['Judul Film', 'Genre']].iloc[movie_indices].reset_index(drop=True)
        return recommended_movies

    def main():
        st.header(" :film_frames: Content Based Filtering ")

        movie_title_to_recommend = st.selectbox(" :anchor: Pilih Judul Film :", df_movies['Judul Film'], index=0, key='search_box')
        num_recommendations = st.slider(" :anchor: Pilih Jumlah Film :", 1, 100, 10)

        if st.button(" :movie_camera: RESULT :bulb:"):
            recommended_movies = recommend_movies(movie_title_to_recommend, similarity_matrix, df_movies, num_recommendations)

            st.subheader(f" :movie_camera: Top {num_recommendations} Rekomendasi Film ")
            st.table(recommended_movies)

    if __name__ == "__main__":
        main()

elif selected == 'Collaborative Filtering':
    st.header(' :film_frames: Collaborative Filtering ')

    user_to_user_encoded = {x: i for i, x in enumerate(movies['userId'].unique())}
    user_encoded_to_user = {i: x for i, x in enumerate(movies['userId'].unique())}
    movie_to_movie_encoded = {x: i for i, x in enumerate(movies['movieId'].unique())}
    movie_encoded_to_movie = {i: x for i, x in enumerate(movies['movieId'].unique())}

    sorted_user_ids = sorted(movies['userId'].unique())

    selected_user = st.selectbox(":anchor: Pilih Pengguna [ UserId ] :", sorted_user_ids)

    if st.button(" :movie_camera: RESULT :bulb: "):
        movie_visited_by_user = movies[movies.userId == selected_user]

        movie_not_visited = movies[~movies['movieId'].isin(movie_visited_by_user.movieId.values)]['movieId']
        movie_not_visited = list(set(movie_not_visited).intersection(set(movie_to_movie_encoded.keys())))
        movie_not_visited = [[movie_to_movie_encoded.get(x)] for x in movie_not_visited]

        unrated_movies_df = pd.DataFrame(movie_not_visited, columns=['movieId'])
        unrated_movies_df['title'] = unrated_movies_df['movieId'].map(movie_encoded_to_movie)

        user_encoder = user_to_user_encoded.get(selected_user)
        user_movie_array = np.hstack(([[user_encoder]] * len(unrated_movies_df), unrated_movies_df[['movieId']].values))

        ratings = model.predict(user_movie_array).flatten()
        recommendations = np.argsort(ratings)[-10:][::-1]

        st.subheader(f" :movie_camera: Rekomendasi Film untuk Users : [{selected_user}] ")
        recommended_movie_ids = [unrated_movies_df.iloc[i]['movieId'] for i in recommendations]
        recommended_movie = movies[movies['movieId'].isin(recommended_movie_ids)][['title', 'genre']]
        st.table(recommended_movie)

        st.markdown("----")
        st.subheader(f" :movie_camera: Film yang Telah Diberikan Ratings oleh Users : [{selected_user}] ")
        all_high_rated_movies = movie_visited_by_user.sort_values(by='rating', ascending=False)
        st.table(all_high_rated_movies[['title', 'genre', 'rating']])
