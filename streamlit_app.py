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

# Загрузка ресурсов NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Предобработка текста
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Токенизация
    filtered = [word for word in tokens if word.isalpha() and word not in stop_words and word not in punctuation]
    return filtered

# Обучение модели Doc2Vec
def train_doc2vec_model(df_en):
    tagged_data = [
        TaggedDocument(words=preprocess_text(desc), tags=[str(i)])
        for i, desc in enumerate(df_en['description'])
    ]

    model = Doc2Vec(
        vector_size=150,   # Размер вектора
        window=5,          # Размер окна
        min_count=2,       # Минимальное количество вхождений слова
        workers=4,         # Количество потоков
        epochs=40,         # Количество эпох
        dm=1,              # Использование модели с direct context
        hs=0,              # Не использовать модели с hierarchical softmax
        negative=10        # Использование негативной выборки
    )

    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    # Сохраняем модель
    model.save("podcast_doc2vec.model")
    return model

# Функция для получения вектора подкаста
def get_podcast_vector(episode_description, model):
    return model.infer_vector(preprocess_text(episode_description))

# Рекомендательная функция
def recommend_similar_podcasts_by_name(episode_name, df_en, model, top_n=20):
    podcast_index = df_en[df_en['episodeName'] == episode_name].index[0]
    target_vector = get_podcast_vector(df_en['description'][podcast_index], model)

    all_vectors = np.array([get_podcast_vector(desc, model) for desc in df_en['description']])

    similarity_scores = cosine_similarity([target_vector], all_vectors)[0]
    similar_indices = similarity_scores.argsort()[-top_n-1:-1][::-1]  # Исключаем сам подкаст

    results = df_en.iloc[similar_indices][['episodeName', 'show.name', 'show.publisher', 'show.total_episodes']].copy()
    results['model_score'] = similarity_scores[similar_indices].round(2)
    
    return results

# Главная страница Streamlit
def main():
    # Загрузка данных и модели
    df_en = load_data()
    model = train_doc2vec_model(df_en)

    # Страница выбора
    st.title("Podcast Recommendation System")
    
    page = st.radio("Choose a page:", ("Home", "Project Info"))
    
    if page == "Home":
        # Показываем топ 20 популярных подкастов
        st.subheader("Top 20 Popular Podcasts")
        top_podcasts = df_en.sort_values(by='rank', ascending=False).head(20)

        # Показываем карточки для топовых подкастов
        for index, row in top_podcasts.iterrows():
            if st.button(f"🎧 {row['episodeName']}"):
                st.write(f"**Episode Name**: {row['episodeName']}")
                st.write(f"**Show Name**: {row['show.name']}")
                st.write(f"**Publisher**: {row['show.publisher']}")
                st.write(f"**Total Episodes**: {row['show.total_episodes']}")
                
                # Получаем похожие подкасты
                similar_podcasts = recommend_similar_podcasts_by_name(row['episodeName'], df_en, model, top_n=20)
                st.write("### Similar Podcasts:")
                
                for _, similar_row in similar_podcasts.iterrows():
                    st.write(f"**Episode**: {similar_row['episodeName']}")
                    st.write(f"**Show**: {similar_row['show.name']}")
                    st.write(f"**Publisher**: {similar_row['show.publisher']}")
                    st.write(f"**Similarity Score**: {similar_row['model_score']}")
    
    if page == "Project Info":
        st.subheader("About the Project")
        st.write("This project is a Podcast Recommendation System built using Doc2Vec, a technique for converting text to vectors. "
                 "The system recommends similar podcasts based on the descriptions of episodes. Users can view top popular podcasts, "
                 "click on them to see more details, and explore recommendations based on their choices.")
        
        st.write("### Technologies Used:")
        st.write("- **Python**")
        st.write("- **Streamlit** for web interface")
        st.write("- **Doc2Vec** for text vectorization")
        st.write("- **Cosine Similarity** for recommendation")
        st.write("- **Pandas** and **NumPy** for data manipulation")

# Запуск приложения
if __name__ == "__main__":
    main()
