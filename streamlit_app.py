import streamlit as st
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

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

class PodcastRecommender:
    def __init__(self, data):
        self.df = data.dropna(subset=['episodeName', 'clean_description'])
        self.df['clean_episodeName'] = self.df['episodeName'].str.lower().str.strip()
        self.df['clean_description'] = self.df['clean_description'].str.lower().str.strip()

        # Train the Doc2Vec model on episode names and descriptions
        self.model = self.train_doc2vec_model()

    def train_doc2vec_model(self):
        documents = []
        for _, row in self.df.iterrows():
            # Create TaggedDocument for episode names and descriptions
            episode_name_doc = TaggedDocument(words=row['clean_episodeName'].split(), tags=[f"episode_{_}"])
            description_doc = TaggedDocument(words=row['clean_description'].split(), tags=[f"description_{_}"])
            documents.extend([episode_name_doc, description_doc])

        # Train the model
        model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=10)
        model.build_vocab(documents)
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
        return model

    def get_similarity(self, s1, s2):
        # Convert strings to vectors using the trained model
        vec1 = self.model.infer_vector(s1.split())
        vec2 = self.model.infer_vector(s2.split())
        return self.model.dv.cosine(vec1, vec2)

    def recommend(self, query, by='title', n=5):
        sim_list = []
        for _, row in self.df.iterrows():
            if by == 'title':
                sim = self.get_similarity(query, row['clean_episodeName'])
            else:
                sim = self.get_similarity(query, row['clean_description'])

            sim_list.append({
                'title': row['episodeName'],
                'description': row['clean_description'],
                'similarity': sim,
                'episodes': row.get('show.total_episodes', 0),
                'rating': row.get('rank', '—'),
                'publisher': row.get('show.publisher', '—'),
                'explicit': row.get('explicit', '—'),
                'duration': row.get('duration_min', '—')
            })
        return sorted(sim_list, key=lambda x: x['similarity'], reverse=True)[:n]

def main():
    st.title("🎧 NextPodcast — Рекомендации по подкастам")

    data = load_podcast_data()
    if data.empty:
        return

    recommender = PodcastRecommender(data)

    st.sidebar.header("🔧 Настройки рекомендаций")
    search_type = st.sidebar.radio("Искать по:", ["Название эпизода", "Описание эпизода"])

    if search_type == "Название эпизода":
        query = st.sidebar.selectbox("Выберите эпизод:", options=data['episodeName'].dropna().unique())
        by = 'title'
    else:
        query = st.sidebar.text_area("Введите описание подкаста:", "")
        by = 'description'

    n_recs = st.sidebar.slider("Количество рекомендаций:", 1, 10, 5)
    show_recs = st.sidebar.button("🔍 Получить рекомендации")

    if not show_recs:
        st.subheader("🔥 Топ-10 популярных подкастов")
        top10 = data.nlargest(10, 'rank')
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
        if query:
            recommendations = recommender.recommend(query, by=by, n=n_recs)
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>{i}. {rec['title']}</h4>
                        <p><strong>Похожесть:</strong> {rec['similarity']:.2f}</p>
                        <p><strong>Рейтинг:</strong> {rec['rating']}</p>
                        <p><strong>Эпизодов:</strong> {rec['episodes']}</p>
                        <p><strong>Издатель:</strong> {rec['publisher']}</p>
                        <p><strong>Эксплицитный:</strong> {rec['explicit']}</p>
                        <p><strong>Длительность:</strong> {rec['duration']} мин</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Пожалуйста, введите запрос для получения рекомендаций.")

if __name__ == "__main__":
    main()
