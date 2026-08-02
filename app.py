import streamlit as st
import requests
import os
import numpy as np
import ollama
from groq import Groq
import yfinance as yf
from dotenv import load_dotenv
from fastembed import TextEmbedding
import streamlit as st

# On Streamlit Cloud, secrets come from st.secrets, not the local .env file.
# This bridges them into os.environ so os.getenv(...) works the same either way.
for key in ["NEWSAPI_KEY", "GROQ_API_KEY", "LLM_PROVIDER"]:
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

load_dotenv()
api_key = os.getenv("NEWSAPI_KEY")


def fetch_news(query, page_size=10):
    params = {
        "q": query,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": page_size,
        "apiKey": api_key,
    }
    response = requests.get("https://newsapi.org/v2/everything", params=params)
    data = response.json()
    return data.get("articles", [])


_embedding_model = TextEmbedding()  # loads once, reused for every call

def get_embedding(text):
    embedding = list(_embedding_model.embed([text]))[0]
    return np.array(embedding)


def get_stock_summary(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        last_price = info.last_price
        previous_close = info.previous_close
        pct_change = ((last_price - previous_close) / previous_close) * 100
        direction = "up" if pct_change >= 0 else "down"
        return f"{ticker} is currently trading at {last_price:.2f}, {direction} {abs(pct_change):.2f}% from previous close of {previous_close:.2f}."
    except Exception as e:
        return f"Live price data for {ticker} is temporarily unavailable ({e}). Continuing with news-based analysis only."
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def rank_articles_by_relevance(question, articles):
    question_vec = get_embedding(question)
    scored_articles = []
    for article in articles:
        text = article["title"] + ". " + (article.get("description") or "")
        article_vec = get_embedding(text)
        score = cosine_similarity(question_vec, article_vec)
        scored_articles.append((score, article))
    scored_articles.sort(key=lambda x: x[0], reverse=True)
    return scored_articles


def run_assistant(ticker, question):
    stock_summary = get_stock_summary(ticker)
    articles = fetch_news(ticker + " stock")
    ranked = rank_articles_by_relevance(question, articles)
    top_articles = ranked[:3]
    context = "\n".join([f"- {article['title']}" for score, article in top_articles])

    prompt = f"""
Current market data:
{stock_summary}

Recent news headlines relevant to {ticker}:
{context}

Based on this data and these headlines, answer the question: {question}
Keep your answer to 3-4 sentences, and don't make up information not given above.
"""
    llm_provider = os.getenv("LLM_PROVIDER", "ollama")

    if llm_provider == "groq":
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
    else:
        response = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])
        answer = response["message"]["content"]

    return stock_summary, top_articles, answer
# --- UI ---

st.title("📈 AI Stock & Finance Assistant")
st.caption("Ask about a stock, grounded in live price data and real news.")

ticker = st.text_input("Ticker symbol", value="AAPL")
question = st.text_input("Your question", value="Why did this stock move this week?")

if st.button("Ask"):
    with st.spinner("Fetching data and thinking..."):
        stock_summary, top_articles, answer = run_assistant(ticker, question)

    st.subheader("Market Data")
    st.write(stock_summary)

    st.subheader("Relevant News")
    for score, article in top_articles:
        st.write(f"**{article['title']}** (relevance: {score:.2f})")

    st.subheader("AI Answer")
    st.write(answer)