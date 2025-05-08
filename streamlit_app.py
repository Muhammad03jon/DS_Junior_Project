import streamlit as st
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Настройка страницы
st.set_page_config(page_title="NextPodcast — Рекомендательная система подкастов", page_icon="🎧", layout="wide")

# Стили
st.markdown("""
    <style>
        .main { padding: 2rem; font-family: 'Open Sans', sans-serif; }
        h1, h2, h3, h4 { color: #4A4A4A; }
        .stButton>button {
            width: 100%; background: linear-gradient(90deg, #6C63FF, #A084DC);
            color: white; border: none; padding: 0.6rem; border-radius: 0.5rem;
            font-weight: bold; transition: 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #4C47E3, #6F5BB5); transform: scale(1.05);
        }
        .recommendation-card {
            padding: 1rem; border-radius: 1rem; background: #F4F4FB;
            margin: 1rem 0; border-left: 5px solid #6C63FF;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05); transition: 0.3s ease;
        }
        .recommendation-card:hover {
            transform: translateY(-4px); box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Загрузка ресурсов
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalpha() and word not in stop_words and word not in punctuation]

@st.cache_data
def load_podcast_data():
    url = "https://raw.githubusercontent.com/Muhammad03jon/DS_Junior_Project/refs/heads/master/data_for_podcasts.csv"
    try:
        df = pd.read_csv(url)
        df['episodeName'] = df['episodeName'].str.strip()
        df['clean_description'] = df['clean_description'].str.strip()
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    return Doc2Vec.load("podcast_doc2vec.model")

def get_podcast_vector(model, description):
    return model.infer_vector(preprocess_text(description))

def recommend_by_doc2vec(model, df, episode_name, top_n=5):
    if episode_name not in df['episodeName'].values:
        return []

    target_desc = df[df['episodeName'] == episode_name]['clean_description'].values[0]
    target_vec = get_podcast_vector(model, target_desc)

    all_vectors = np.array([get_podcast_vector(model, desc) for desc in df['clean_description']])
    similarities = cosine_similarity([target_vec], all_vectors)[0]
    similar_indices = similarities.argsort()[-top_n-1:-1][::-1]

    results = df.iloc[similar_indices].copy()
    results['similarity'] = similarities[similar_indices].round(2)
    return results[['episodeName', 'rank', 'show.publisher', 'show.total_episodes', 'explicit', 'duration_min', 'similarity']]

def main():
    st.title("🎧 NextPodcast — Рекомендации по подкастам")

    df = load_podcast_data()
    if df.empty:
        return

    model = load_model()

    st.sidebar.header("🔧 Настройки рекомендаций")
    episode_name = st.sidebar.selectbox("Выберите эпизод:", options=df['episodeName'].dropna().unique())
    n_recs = st.sidebar.slider("Количество рекомендаций:", 1, 10, 5)
    show_recs = st.sidebar.button("🔍 Получить рекомендации")

    if not show_recs:
        st.subheader("🔥 Топ-10 популярных подкастов")
        top10 = df.nlargest(10, 'rank')
        cols = st.columns(2)
        for i, (_, pod) in enumerate(top10.iterrows()):
            with cols[i % 2]:
                st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>{pod['episodeName']}</h4>
                        <p><strong>Рейтинг:</strong> {pod['rank']}</p>
                        <p><strong>Издатель:</strong> {pod.get('show.publisher', '—')}</p>
                        <p><strong>Эпизодов:</strong> {pod.get('show.total_episodes', '—')}</p>
                        <p><strong>Длительность:</strong> {pod.get('duration_min', '—')} мин</p>
                        <p><strong>Эксплицитный:</strong> {pod.get('explicit', '—')}</p>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.subheader("🎯 Рекомендованные подкасты")
        recommendations = recommend_by_doc2vec(model, df, episode_name, n_recs)
        if recommendations.empty:
            st.warning("Нет рекомендаций.")
        else:
            for i, (_, rec) in enumerate(recommendations.iterrows(), 1):
                st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>{i}. {rec['episodeName']}</h4>
                        <p><strong>Похожесть:</strong> {rec['similarity']}</p>
                        <p><strong>Рейтинг:</strong> {rec['rank']}</p>
                        <p><strong>Издатель:</strong> {rec['show.publisher']}</p>
                        <p><strong>Эпизодов:</strong> {rec['show.total_episodes']}</p>
                        <p><strong>Эксплицитный:</strong> {rec['explicit']}</p>
                        <p><strong>Длительность:</strong> {rec['duration_min']} мин</p>
                    </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
