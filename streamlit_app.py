import streamlit as st
import pandas as pd
import numpy as np
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity

# Загрузка данных и модели
url = "https://raw.githubusercontent.com/Muhammad03jon/DS_Junior_Project/refs/heads/master/data_for_podcasts.csv"
df_en = pd.read_csv(url)  # Замените на путь к вашему файлу
model = Doc2Vec.load("podcast_doc2vec.model")

# Функции
def preprocess_text(text):
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import string

    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalpha() and word not in stop_words and word not in punctuation]

def get_podcast_vector(description):
    return model.infer_vector(preprocess_text(description))

def recommend_similar_podcasts_by_index(index, top_n=20):
    target_vector = get_podcast_vector(df_en['clean_description'][index])
    all_vectors = np.array([get_podcast_vector(desc) for desc in df_en['clean_description']])
    similarity_scores = cosine_similarity([target_vector], all_vectors)[0]
    similar_indices = similarity_scores.argsort()[-top_n-1:-1][::-1]
    results = df_en.iloc[similar_indices].copy()
    results["similarity"] = similarity_scores[similar_indices]
    return results

# Интерфейс Streamlit
st.set_page_config(page_title="Podcast Recommender", layout="wide")

# Состояние
if "selected_index" not in st.session_state:
    st.session_state.selected_index = None

# Верхнее меню
menu = st.sidebar.selectbox("Навигация", ["🏠 Главная", "📄 Презентация"])

if menu == "🏠 Главная":
    st.title("🎧 Discover Popular Podcasts")

    if st.session_state.selected_index is None:
        st.subheader("🔥 Top 20 Popular Podcasts")
        top_podcasts = df_en.sort_values("rank").head(20)

        cols = st.columns(2)
        for i, row in top_podcasts.iterrows():
            with cols[i % 2]:
                if st.button(f"{row['episodeName']} ({row['show.name']})", key=f"podcast_{i}"):
                    st.session_state.selected_index = i
                    st.experimental_rerun()
    else:
        # Детали подкаста
        row = df_en.iloc[st.session_state.selected_index]
        st.subheader(row['episodeName'])
        st.markdown(f"**🎙️ Show:** {row['show.name']}")
        st.markdown(f"**📢 Publisher:** {row['show.publisher']}")
        st.markdown(f"**🎬 Episodes:** {row['show.total_episodes']}")
        st.markdown(f"**🕐 Duration (min):** {row['duration_min']:.2f}")
        st.markdown(f"**📝 Description:** {row['description']}")

        st.divider()
        st.subheader("🔁 Recommended Podcasts")
        recs = recommend_similar_podcasts_by_index(st.session_state.selected_index)

        cols = st.columns(2)
        for i, rec_row in recs.iterrows():
            with cols[i % 2]:
                if st.button(f"{rec_row['episodeName']} ({rec_row['show.name']})", key=f"rec_{i}"):
                    st.session_state.selected_index = rec_row.name
                    st.experimental_rerun()

        if st.button("⬅️ Back to Top 20"):
            st.session_state.selected_index = None
            st.experimental_rerun()

elif menu == "📄 Презентация":
    st.title("📄 Project Overview")
    st.markdown("""
    ## 🎧 Podcast Recommender System
    This project recommends podcasts based on description similarity using Doc2Vec and cosine similarity.

    ### 🔧 Technologies Used
    - Python, Pandas, Streamlit
    - Gensim Doc2Vec
    - Cosine Similarity from scikit-learn

    ### 💡 Features
    - Shows Top 20 Popular Podcasts
    - On click, shows similar podcasts sorted by similarity
    - Clean interface with intuitive navigation

    [🔗 GitHub Repository](https://github.com/yourusername/yourrepo)
    """)

