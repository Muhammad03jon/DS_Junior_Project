import streamlit as st
import pandas as pd
import numpy as np
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity

# Настройка страницы
st.set_page_config(page_title="NextPodcast — Рекомендательная система подкастов", page_icon="🎧", layout="wide")

# Загрузка ресурсов
nltk.download('punkt')
nltk.download('stopwords')

# Предобработка текста
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

def preprocess_text(text):
    tokens = word_tokenize(str(text).lower())
    filtered = [word for word in tokens if word.isalpha() and word not in stop_words and word not in punctuation]
    return filtered

# Загрузка данных
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Muhammad03jon/DS_Junior_Project/refs/heads/master/data_for_podcasts.csv"
    df = pd.read_csv(url)
    df['clean_description'] = df['clean_description'].fillna('').astype(str)
    df['episodeName'] = df['episodeName'].fillna('').astype(str)
    return df

df = load_data()

# Подготовка данных для Doc2Vec
@st.cache_resource
def train_doc2vec_model(df):
    tagged_data = [
        TaggedDocument(words=preprocess_text(desc), tags=[str(i)])
        for i, desc in enumerate(df['clean_description'])
    ]
    model = Doc2Vec(
        vector_size=150,
        window=5,
        min_count=2,
        workers=4,
        epochs=40,
        dm=1,
        hs=0,
        negative=10
    )
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    return model

model = train_doc2vec_model(df)

# Вычисляем векторы всех описаний заранее
@st.cache_data
def compute_all_vectors():
    return np.array([model.infer_vector(preprocess_text(desc)) for desc in df['clean_description']])

all_vectors = compute_all_vectors()

# Рекомендательная функция
def recommend_similar_podcasts(episode_name, top_n=10):
    try:
        idx = df[df['episodeName'] == episode_name].index[0]
    except IndexError:
        return pd.DataFrame()
    
    target_vector = all_vectors[idx].reshape(1, -1)
    similarity_scores = cosine_similarity(target_vector, all_vectors)[0]
    similar_indices = similarity_scores.argsort()[-top_n-1:-1][::-1]

    results = df.iloc[similar_indices][[
        'episodeName', 'show.name', 'show.publisher', 'show.total_episodes',
        'explicit', 'duration_min'
    ]].copy()
    results['similarity'] = similarity_scores[similar_indices].round(3)
    return results.reset_index(drop=True)

# Интерфейс
st.title("🎧 NextPodcast — Рекомендательная система на основе Doc2Vec")

selected_episode = st.selectbox("Выберите подкаст для рекомендации:", df['episodeName'].unique())

if st.button("Показать рекомендации"):
    recommendations = recommend_similar_podcasts(selected_episode, top_n=10)
    if recommendations.empty:
        st.warning("Подкаст не найден или отсутствует описание.")
    else:
        st.subheader(f"🔍 Похожие на: **{selected_episode}**")
        for i, row in recommendations.iterrows():
            with st.container():
                if st.button(f"{i+1}. {row['episodeName']']}", key=f"rec_btn_{i}"):
                    st.markdown(f"""
                        <div style="padding: 1rem; border-left: 5px solid #6C63FF; background: #f0f0fa; border-radius: 10px; margin: 1rem 0;">
                            <h4>{row['episodeName']}</h4>
                            <p><strong>Шоу:</strong> {row['show.name']}</p>
                            <p><strong>Издатель:</strong> {row['show.publisher']}</p>
                            <p><strong>Эпизодов:</strong> {row['show.total_episodes']}</p>
                            <p><strong>Длительность:</strong> {row['duration_min']} мин</p>
                            <p><strong>Explicit:</strong> {row['explicit']}</p>
                            <p><strong>Похожесть:</strong> {row['similarity']}</p>
                        </div>
                    """, unsafe_allow_html=True)

                    # Показываем ещё похожие на этот подкаст
                    st.markdown(f"### 🎯 Похожие на «{row['episodeName']}»")
                    more_recs = recommend_similar_podcasts(row['episodeName'], top_n=5)
                    for j, sim in more_recs.iterrows():
                        st.markdown(f"- **{sim['episodeName']}** (похожесть: {sim['similarity']})")
