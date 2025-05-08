import streamlit as st
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

# Настройка страницы
st.set_page_config(page_title="Рекомендательная система подкастов", page_icon="🎧", layout="wide")

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
        df['episodeName'] = df['episodeName'].astype(str).str.strip()
        df['clean_description'] = df['clean_description'].astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return pd.DataFrame()

class PodcastRecommender:
    def __init__(self, data):
        self.df = data.dropna(subset=['episodeName', 'clean_description'])
        self.df['clean_episodeName'] = self.df['episodeName'].str.lower().str.strip()
        self.df['clean_description'] = self.df['clean_description'].str.lower().str.strip()

        tagged_data = [
            TaggedDocument(words=doc.split(), tags=[str(i)])
            for i, doc in enumerate(self.df['clean_description'])
        ]
        self.model = Doc2Vec(vector_size=100, window=2, min_count=1, workers=4)
        self.model.build_vocab(tagged_data)
        self.model.train(tagged_data, total_examples=self.model.corpus_count, epochs=10)

    def recommend(self, episode_title, n=5):
        try:
            query_desc = self.df[self.df['episodeName'] == episode_title]['clean_description'].values[0]
        except IndexError:
            return []

        inferred_vector = self.model.infer_vector(query_desc.split())
        sims = self.model.dv.most_similar([inferred_vector], topn=n + 1)

        results = []
        for tag, sim_score in sims:
            index = int(tag)
            row = self.df.iloc[index]
            if row['episodeName'] == episode_title:
                continue  # не включаем сам эпизод
            results.append({
                'title': row['episodeName'],
                'description': row['clean_description'],
                'episodes': row.get('show.total_episodes', 0),
                'rating': row.get('rank', '—'),
                'publisher': row.get('show.publisher', '—'),
                'explicit': row.get('explicit', '—'),
                'duration': row.get('duration_min', '—')
            })
            if len(results) >= n:
                break
        return results

def main():
    st.title("Рекомендательная система подкастов")

    data = load_podcast_data()
    if data.empty:
        return

    recommender = PodcastRecommender(data)

    st.sidebar.header("🔧 Настройки рекомендаций")
    query = st.sidebar.selectbox("Выберите эпизод:", options=data['episodeName'].dropna().unique())
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
            recommendations = recommender.recommend(query, n=n_recs)
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>{i}. {rec['title']}</h4>
                        <p><strong>Рейтинг:</strong> {rec['rating']}</p>
                        <p><strong>Эпизодов:</strong> {rec['episodes']}</p>
                        <p><strong>Издатель:</strong> {rec['publisher']}</p>
                        <p><strong>Эксплицитный:</strong> {rec['explicit']}</p>
                        <p><strong>Длительность:</strong> {rec['duration']} мин</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Пожалуйста, выберите эпизод для получения рекомендаций.")

if __name__ == "__main__":
    main()
