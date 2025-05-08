import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Загрузка ресурсов NLTK
nltk.download('stopwords')

# Простой токенизатор с использованием регулярных выражений
def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# Предобработка текста
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = simple_tokenize(text)
    filtered = [word for word in tokens if word.isalpha() and word not in stop_words]
    return filtered

# Загрузка модели
model = Doc2Vec.load("podcast_doc2vec.model")

# Функция для получения вектора подкаста
def get_podcast_vector(episode_description):
    return model.infer_vector(preprocess_text(episode_description))

# Функция для получения индекса по названию эпизода
def get_podcast_index_by_name(episode_name, df):
    return df[df['episodeName'] == episode_name].index[0]

# Рекомендательная функция
def recommend_similar_podcasts_by_name(episode_name, df, top_n=20):
    podcast_index = get_podcast_index_by_name(episode_name, df)
    target_vector = get_podcast_vector(df['clean_description'][podcast_index])

    # Вычисление всех векторов заранее
    all_vectors = np.array([get_podcast_vector(desc) for desc in df['clean_description']])

    similarity_scores = cosine_similarity([target_vector], all_vectors)[0]
    similar_indices = similarity_scores.argsort()[-top_n-1:-1][::-1]  # -1 исключает сам подкаст

    results = df.iloc[similar_indices][['episodeName', 'show.name', 'show.publisher', 'show.total_episodes']].copy()
    results['model_index'] = similar_indices
    results['model_score'] = similarity_scores[similar_indices].round(2)

    return results.reset_index(drop=True)

# Streamlit интерфейс
def main():
    # Загружаем данные
    df = pd.read_csv('your_dataset.csv')  # Путь к вашему набору данных

    st.title('Podcast Recommendation System')
    episode_name = st.text_input('Enter the name of the podcast episode:')
    
    if episode_name:
        top_n = st.slider('Select number of recommendations:', 1, 20, 5)
        
        st.write("Generating recommendations...")
        
        recommendations = recommend_similar_podcasts_by_name(episode_name, df, top_n)
        
        if not recommendations.empty:
            st.write(f"Top {top_n} similar podcasts to '{episode_name}':")
            st.dataframe(recommendations)
        else:
            st.write("No recommendations found for this episode.")

if __name__ == "__main__":
    main()
