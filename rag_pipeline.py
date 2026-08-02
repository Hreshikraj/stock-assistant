import requests
import os
import numpy as np
import ollama
import yfinance as yf
from dotenv import load_dotenv

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


def get_embedding(text):
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return np.array(response["embedding"])


def get_stock_summary(ticker):
    t = yf.Ticker(ticker)
    info = t.fast_info
    last_price = info.last_price
    previous_close = info.previous_close
    pct_change = ((last_price - previous_close) / previous_close) * 100
    direction = "up" if pct_change >= 0 else "down"
    return f"{ticker} is currently trading at {last_price:.2f}, {direction} {abs(pct_change):.2f}% from previous close of {previous_close:.2f}."


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


# --- Main script ---

articles = fetch_news("Apple stock")
print(f"Fetched {len(articles)} articles")

question = "Why did Apple stock move this week?"
ranked = rank_articles_by_relevance(question, articles)

print(f"\nTop matches for: '{question}'\n")
for score, article in ranked[:3]:
    print(f"{score:.3f} - {article['title']}")

stock_summary = get_stock_summary("AAPL")

top_articles = ranked[:3]
context = "\n".join([f"- {article['title']}" for score, article in top_articles])

prompt = f"""
Current market data:
{stock_summary}

Recent news headlines relevant to Apple stock:
{context}

Based on this data and these headlines, answer the question: {question}
Keep your answer to 3-4 sentences, and don't make up information not given above.
"""

response = ollama.chat(
    model="llama3.1",
    messages=[{"role": "user", "content": prompt}]
)

print("\n--- AI Answer ---")
print(response["message"]["content"])