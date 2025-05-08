import streamlit as st
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity

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

# Загрузка данных
@st.cache_data
def load_podcast_data():
    url = "https://raw.githubusercontent.com/Muhammad03jon/DS_Junior_Project/refs/heads/master/data_for_podcasts.csv"
    try:
        data = pd.read_csv(url)
        data['episodeName'] = data['episodeName'].str.strip()
        data['clean_description'] = data['clean_description'].str.strip()
        return data.dropna(subset=['episodeName', 'clean_description'])
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return pd.DataFrame()

# Кэшируем модель Doc2Vec
@st.cache_resource
def train_doc2vec_model(df, column='clean_description'):
    documents = [TaggedDocument(words=row[column].lower().split(), tags=[str(i)]) for i, row in df.iterrows()]
    model = Doc2Vec(vector_size=100, min_count=2, epochs=40, workers=4, seed=42)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    return model

# Класс рекомендаций
class PodcastRecommender:
    def __init__(self, data, model):
        self.df = data.reset_index(drop=True)
        self.model = model
        self.embeddings = np.array([model.infer_vector(desc.lower().split()) for desc in self.df['clean_description']])

    def recommend(self, query, by='description', n=5):
        if by == 'title':
            match = self.df[self.df['episodeName'] == query]
            if match.empty:
                return []
            vector = self.model.infer_vector(match.iloc[0]['episodeName'].lower().split())
        else:
            vector = self.model.infer_vector(query.lower().split())

        sims = cosine_similarity([vector], self.embeddings)[0]
        self.df['similarity'] = sims
        results = self.df.sort_values(by='similarity', ascending=False).head(n)
        return results.to_dict('records')

# Основная логика
def main():
    st.title("🎧 NextPodcast — Рекомендательная система для подкастов")

    data = load_podcast_data()
    if data.empty:
        st.stop()

    model = train_doc2vec_model(data)
    recommender = PodcastRecommender(data, model)

    st.sidebar.header("Настройки рекомендаций")

    with st.expander("Поиск по названию или описанию подкаста"):
        search_type = st.radio("Искать по:", ["Название эпизода", "Описание эпизода"])
        by = 'title' if search_type == "Название эпизода" else 'description'

        if by == 'title':
            query = st.selectbox("Выберите эпизод:", options=data['episodeName'].unique())
        else:
            query = st.text_input("Введите описание эпизода:")

        n_recs = st.slider("Количество рекомендаций:", 1, 10, 5)
        show_recs = st.button("Получить рекомендации")

    if show_recs and query:
        recommendations = recommender.recommend(query, by=by, n=n_recs)
        st.subheader("🔍 Рекомендованные подкасты:")

        for i, rec in enumerate(recommendations):
            with st.container():
                key = f"rec_{i}"
                if st.button(f"{i+1}. {rec['episodeName']}", key=key):
                    st.markdown(f"""
                        <div class="recommendation-card">
                            <h3>{rec['episodeName']}</h3>
                            <p><strong>Описание:</strong> {rec['clean_description']}</p>
                            <p><strong>Оценка:</strong> {rec.get('rank', 'N/A')}</p>
                            <p><strong>Эпизодов:</strong> {rec.get('show.total_episodes', 0)}</p>
                            <p><strong>Издатель:</strong> {rec['show.publisher']}</p>
                            <p><strong>Эксплицитный:</strong> {rec['explicit']}</p>
                            <p><strong>Длительность (мин.):</strong> {rec['duration_min']}</p>
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"### 🎯 Похожие на «{rec['episodeName']}»")
                    similar_recs = recommender.recommend(rec['clean_description'], by='description', n=10)
                    for j, sim_rec in enumerate(similar_recs[1:], 1):
                        st.markdown(f"**{j}. {sim_rec['episodeName']}** — Оценка: {sim_rec.get('rank', 'N/A')}, Похожесть: {sim_rec['similarity']:.2f}")

if __name__ == "__main__":
    main()
