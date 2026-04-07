import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
import re
import numpy as np
import time
import pandas as pd

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# 🔥 NLTK FIX (ALL REQUIRED)
# ---------------------------
def download_nltk():
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng')
    ]

    for path, name in resources:
        try:
            nltk.data.find(path)
        except:
            nltk.download(name)

download_nltk()

# ---------------------------
# 🔹 POS TAG CONVERTER
# ---------------------------
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

# ---------------------------
# 🔹 PREPROCESS WITH LEMMATIZATION
# ---------------------------
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    processed_tokens = []
    for word, tag in tagged:
        if word not in stop_words and len(word) > 2:
            lemma = lemmatizer.lemmatize(word, get_wordnet_pos(tag))
            processed_tokens.append(lemma)

    return processed_tokens[:500]

# ---------------------------
# 🔹 JACCARD SIMILARITY
# ---------------------------
def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if len(a | b) != 0 else 0

# ---------------------------
# 🔹 CLEAN TEXT EXTRACTION
# ---------------------------
def extract_text(soup):
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    article = soup.find("article")

    if article:
        paragraphs = article.find_all("p")
    else:
        paragraphs = soup.find_all("p")

    texts = []
    for p in paragraphs:
        t = p.get_text().strip().lower()

        if (
            len(t) > 50 and
            "bezzy" not in t and
            "subscribe" not in t and
            "advertisement" not in t
        ):
            texts.append(t)

    return " ".join(texts[:50])

# ---------------------------
# 🔹 WEB CRAWLER
# ---------------------------
def crawl(urls):
    docs = []
    valid = []

    for i, url in enumerate(urls):
        try:
            if i > 0:
                time.sleep(1)

            headers = {"User-Agent": "Mozilla/5.0"}
            res = requests.get(url, headers=headers, timeout=10)
            res.raise_for_status()

            soup = BeautifulSoup(res.text, "html.parser")
            text = extract_text(soup)

            if len(text.split()) > 100:
                docs.append(text)
                valid.append(url)
                st.success(f"✅ Loaded: {url}")
            else:
                st.warning(f"⚠️ Low content: {url}")

        except Exception as e:
            st.error(f"❌ Failed: {url}")

    return docs, valid

# ---------------------------
# 🔹 STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="URL Similarity Analyzer", layout="wide")

st.title("🌐 URL Content Similarity & Clustering (Final Version with Lemmatization)")

urls_input = st.text_area("Enter URLs (one per line):")

if st.button("Analyze"):

    urls = [u.strip() for u in urls_input.split("\n") if u.strip()]
    docs, valid_urls = crawl(urls)

    if len(docs) < 2:
        st.error("❌ Need at least 2 valid URLs")
        st.stop()

    # ---------------------------
    # 🔹 PREPROCESS
    # ---------------------------
    token_docs = [preprocess(doc) for doc in docs]
    cleaned_docs = [" ".join(tokens) for tokens in token_docs]

    # ---------------------------
    # 🔹 TF-IDF
    # ---------------------------
    vectorizer = TfidfVectorizer(
        max_features=1500,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(cleaned_docs)
    tfidf_sim = cosine_similarity(X)

    # ---------------------------
    # 🔥 TOP WORDS + TF-IDF
    # ---------------------------
    st.subheader("🏷️ Top Words with TF-IDF Scores")

    feature_names = vectorizer.get_feature_names_out()

    for idx, url in enumerate(valid_urls):
        row = X[idx].toarray().flatten()
        top_indices = row.argsort()[-10:][::-1]

        words = [feature_names[i] for i in top_indices if row[i] > 0]
        scores = [row[i] for i in top_indices if row[i] > 0]

        df = pd.DataFrame({
            "Word": words,
            "TF-IDF Score": [round(s, 3) for s in scores]
        })

        short_url = url.replace("https://", "").replace("http://", "")[:50]

        with st.expander(f"📄 {short_url}"):
            st.dataframe(df, use_container_width=True)

    # ---------------------------
    # 🔍 SIMILARITY RESULTS
    # ---------------------------
    st.subheader("🔍 Similarity Results")

    scores = []

    for i in range(len(valid_urls)):
        for j in range(i + 1, len(valid_urls)):

            tfidf_score = tfidf_sim[i][j]
            jaccard_score = jaccard(token_docs[i], token_docs[j])

            final_score = (tfidf_score + jaccard_score) / 2
            scores.append(final_score)

            name1 = valid_urls[i].split("//")[1].split("/")[0]
            name2 = valid_urls[j].split("//")[1].split("/")[0]

            if final_score > 0.3:
                st.success(f"✅ {name1} ↔ {name2} = {final_score:.3f} (Similar)")
            elif final_score > 0.15:
                st.warning(f"⚠️ {name1} ↔ {name2} = {final_score:.3f} (Moderate)")
            else:
                st.info(f"ℹ️ {name1} ↔ {name2} = {final_score:.3f} (Not Similar)")

    # ---------------------------
    # 📈 SUMMARY
    # ---------------------------
    st.subheader("📈 Summary")

    st.metric("URLs Processed", len(valid_urls))
    st.metric("Average Similarity", f"{np.mean(scores):.3f}")
