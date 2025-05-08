import streamlit as st
import pandas as pd
from difflib import SequenceMatcher
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

# Загрузка и настройка страницы
st.set_page_config(page_title="NextPodcast — Рекомендательная система подкастов", page_icon="🎧", layout="wide")

# Стиль страницы
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=Open+Sans:wght@400;600&display=swap');
        .main { padding: 2rem; font-family: 'Open Sans', sans-serif; }
        h1, h2, h3, h4, h5 { font-family: 'Roboto', sans-serif; color: #4A4A4A; }
        .stButton>button { width: 100%; background: linear-gradient(90deg, #6C63FF, #A084DC); color: white; border: none; padding: 0.6rem; border-radius: 0.5rem; font-weight: bold; transition: background 0.3s ease, transform 0.3s ease; }
        .stButton>button:hover { background: linear-gradient(90deg, #4C47E3, #6F5BB5); transform: scale(1.05); color: white; }
        .recommendation-card { padding: 1.5rem; border-radius: 1rem; background: linear-gradient(135deg, #F0F0F5, #D9D9E4); margin: 1rem 0; border-left: 5px solid #6C63FF; box-shadow: 0 4px 8px rgba(0,0,0,0.1); color: #333333; transition: transform 0.3s ease, box-shadow 0.3s ease; }
        .recommendation-card:hover { transform: translateY(-5px); box-shadow: 0 8px 16px rgba(0,0,0,0.15); }
    </style>
""", unsafe_allow_html=True)

# Загрузка данных подкастов
@st.cache_data
def load_podcast_data():
    url = "https://raw.githubusercontent.com/Muhammad03jon/DS_Junior_Project/refs/heads/master/data_for_podcasts.csv"  # Здесь должен быть URL с данными
    try:
        data = pd.read_csv(url)
        data['title'] = data['title'].str.strip()
        data['genres'] = data['genres'].str.strip()
        data['description'] = data['description'].str.strip()
        return data
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return pd.DataFrame()

class PodcastRecommender:
    def __init__(self, data):
        self.df = pd.DataFrame(data)
        self.df = self.df.dropna(subset=['title', 'description', 'genres'])
        self.df['clean_title'] = self.df['title'].str.lower().str.strip()
        self.df['clean_description'] = self.df['description'].str.lower().str.strip()

    def get_title_similarity(self, title1, title2):
        return SequenceMatcher(None, title1.lower(), title2.lower()).ratio()

    def get_description_similarity(self, desc1, desc2):
        return SequenceMatcher(None, desc1.lower(), desc2.lower()).ratio()

    def recommend_podcasts(self, query, by='title', n_recommendations=5):
        similarities = []
        if by == 'title':
            for idx, row in self.df.iterrows():
                similarity = self.get_title_similarity(query, row['clean_title'])
                similarities.append(self._create_recommendation_dict(row, similarity))
        elif by == 'description':
            for idx, row in self.df.iterrows():
                similarity = self.get_description_similarity(query, row['clean_description'])
                similarities.append(self._create_recommendation_dict(row, similarity))

        return sorted(similarities, key=lambda x: x['similarity'], reverse=True)[:n_recommendations]

    def _create_recommendation_dict(self, row, similarity):
        return {
            'podcastID': row['id'],
            'title': row['title'],
            'description': row['description'],
            'similarity': similarity,
            'genres': row['genres'],
            'episodes_count': row.get('episodes_count', 0),
            'average_rating': row.get('average_rating', 'N/A')
        }

def main():
    st.title("🎧 NextPodcast — Рекомендательная система для подкастов")
    st.markdown("""
    **NextPodcast** — это интеллектуальная рекомендательная система, которая помогает пользователю найти новый интересный подкаст на основе:
    - Названия подкаста 🎤
    - Описания подкаста 📝
    """)

    data = load_podcast_data()
    if data.empty:
        st.error("Не удалось загрузить данные. Проверь URL CSV.")
        return

    recommender = PodcastRecommender(data)

    # Главная страница — популярные подкасты
    st.sidebar.header("Настройки рекомендаций")

    # Показываем топ 50 популярных подкастов
    top_podcasts = data.nlargest(50, 'average_rating')  # или любой другой критерий, например, по количеству эпизодов
    st.subheader("Топ 50 популярных подкастов")
    for i, podcast in top_podcasts.iterrows():
        st.markdown(f"**{podcast['title']}** - {podcast['average_rating']} ⭐")

    # Получаем рекомендацию
    search_type = st.sidebar.radio("Искать по:", ["Название подкаста", "Описание подкаста"])

    if search_type == "Название подкаста":
        query = st.sidebar.selectbox("Выберите подкаст:", options=data['title'].unique())
        by = 'title'
    else:
        query = st.sidebar.text_input("Введите описание подкаста:", "")
        by = 'description'

    n_recommendations = st.sidebar.slider("Количество рекомендаций:", min_value=1, max_value=10, value=5)

    if st.sidebar.button("Получить рекомендации"):
        recommendations = recommender.recommend_podcasts(query, by, n_recommendations)

        for i, podcast in enumerate(recommendations, 1):
            st.markdown(f"""
            <div class="recommendation-card">
                <h3>{i}. {podcast['title']}</h3>
                <p><strong>Жанры:</strong> {podcast['genres']}</p>
                <p><strong>Похожесть:</strong> {podcast['similarity']:.2f}</p>
                <p><strong>Оценка:</strong> {podcast['average_rating']}</p>
                <p><strong>Количество эпизодов:</strong> {podcast['episodes_count']}</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
