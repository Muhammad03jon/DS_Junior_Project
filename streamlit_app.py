import streamlit as st
import pandas as pd
import gensim
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Настройка страницы
st.set_page_config(page_title="NextPodcast — Рекомендательная система подкастов", page_icon="🎧", layout="wide")

# Стилизация
st.markdown("""
    <style>
        .main { padding: 2rem; font-family: 'Open Sans', sans-serif; }
        h1, h2, h3, h4, h5 { font-family: 'Roboto', sans-serif; color: #4A4A4A; }
        .recommendation-card {
            padding: 1.5rem; border-radius: 1rem;
            background: linear-gradient(135deg, #F0F0F5, #D9D9E4);
            margin: 1rem 0; border-left: 5px solid #6C63FF;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1); color: #333;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            cursor: pointer;
        }
        .recommendation-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        }
    </style>
""", unsafe_allow_html=True)

# Загрузка данных и модели
@st.cache_data
def load_data_and_model():
    url = "https://raw.githubusercontent.com/Muhammad03jon/DS_Junior_Project/refs/heads/master/data_for_podcasts.csv"
    try:
        data = pd.read_csv(url)
        data['episodeName'] = data['episodeName'].astype(str).str.strip()
        data['clean_description'] = data['clean_description'].astype(str).str.strip()
        model = Doc2Vec.load("https://github.com/Muhammad03jon/DS_Junior_Project/blob/master/podcast_doc2vec.model")  # Укажи путь к своей модели
        return data, model
    except Exception as e:
        st.error(f"Ошибка загрузки данных или модели: {e}")
        return pd.DataFrame(), None

# Класс рекомендаций
class PodcastRecommender:
    def __init__(self, data, model):
        self.df = data.dropna(subset=['episodeName', 'clean_description']).reset_index(drop=True)
        self.model = model
        self.vectors = np.array([model.infer_vector(row.split()) for row in self.df['clean_description']])

    def recommend(self, query, by='title', n=5):
        if by == 'title':
            selected = self.df[self.df['episodeName'] == query]
            if selected.empty:
                return []
            vector = self.model.infer_vector(selected.iloc[0]['clean_description'].split())
        else:
            vector = self.model.infer_vector(query.split())

        sims = cosine_similarity([vector], self.vectors)[0]
        indices = sims.argsort()[::-1][:n + 1]
        recs = []
        for idx in indices:
            item = self.df.iloc[idx]
            sim_score = sims[idx]
            recs.append({
                'title': item['episodeName'],
                'description': item['clean_description'],
                'similarity': sim_score,
                'episodes_count': item.get('show.total_episodes', 0),
                'average_rating': item.get('rank', 'N/A'),
                'publisher': item['show.publisher'],
                'explicit': item['explicit'],
                'duration': item['duration_min'],
            })
        return recs[1:n+1]  # Исключаем сам подкаст

# Основная логика
def main():
    st.title("🎧 NextPodcast — Рекомендательная система для подкастов")

    data, model = load_data_and_model()
    if data.empty or model is None:
        st.stop()

    recommender = PodcastRecommender(data, model)

    st.sidebar.header("Настройки рекомендаций")
    search_type = st.sidebar.radio("Искать по:", ["Название эпизода", "Описание эпизода"])
    by = 'title' if search_type == "Название эпизода" else 'description'

    if by == 'title':
        query = st.sidebar.selectbox("Выберите эпизод:", options=data['episodeName'].unique())
    else:
        query = st.sidebar.text_area("Введите описание эпизода:")

    n_recs = st.sidebar.slider("Количество рекомендаций:", 1, 10, 5)
    show_recs = st.sidebar.button("Получить рекомендации")

    if show_recs and query:
        recommendations = recommender.recommend(query, by=by, n=n_recs)
        st.subheader("🔍 Рекомендованные подкасты:")
        for i, rec in enumerate(recommendations):
            with st.container():
                key = f"rec_{i}"
                if st.button(f"{i+1}. {rec['title']}", key=key):
                    st.markdown(f"""
                        <div class="recommendation-card">
                            <h3>{rec['title']}</h3>
                            <p><strong>Описание:</strong> {rec['description']}</p>
                            <p><strong>Оценка:</strong> {rec['average_rating']}</p>
                            <p><strong>Эпизодов:</strong> {rec['episodes_count']}</p>
                            <p><strong>Издатель:</strong> {rec['publisher']}</p>
                            <p><strong>Эксплицитный:</strong> {rec['explicit']}</p>
                            <p><strong>Длительность (мин.):</strong> {rec['duration']}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown(f"### 🎯 Похожие на «{rec['title']}»")
                    similar_recs = recommender.recommend(rec['title'], by='title', n=10)
                    for j, sim_rec in enumerate(similar_recs, 1):
                        st.markdown(f"**{j}. {sim_rec['title']}** — Оценка: {sim_rec['average_rating']}, Похожесть: {sim_rec['similarity']:.2f}")

if __name__ == "__main__":
    main()
