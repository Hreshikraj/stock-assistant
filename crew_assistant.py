import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
import yfinance as yf
import requests
import numpy as np
from fastembed import TextEmbedding

load_dotenv()

api_key = os.getenv("NEWSAPI_KEY")
_embedding_model = TextEmbedding()

llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
)

market_analyst = Agent(
    role="Market Analyst",
    goal="Report on current stock price and trend, based only on data given",
    backstory="A precise financial analyst who only reports verified numbers, never guesses.",
    llm=llm,
)

news_analyst = Agent(
    role="News Analyst",
    goal="Summarize relevant news headlines and their likely impact on the stock",
    backstory="A financial journalist who identifies the most relevant news and explains its significance.",
    llm=llm,
)

advisor = Agent(
    role="Financial Advisor",
    goal="Combine market data and news into one clear, balanced answer for the user",
    backstory="An advisor who synthesizes information clearly and always includes a brief disclaimer that this isn't financial advice.",
    llm=llm,
)


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
        return f"Live price data for {ticker} is temporarily unavailable ({e})."


def fetch_news(query, page_size=10):
    params = {
        "q": query, "sortBy": "publishedAt", "language": "en",
        "pageSize": page_size, "apiKey": api_key,
    }
    response = requests.get("https://newsapi.org/v2/everything", params=params)
    return response.json().get("articles", [])


def get_embedding(text):
    return np.array(list(_embedding_model.embed([text]))[0])


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_top_news(question, ticker, top_n=3):
    articles = fetch_news(ticker + " stock")
    question_vec = get_embedding(question)
    scored = []
    for article in articles:
        text = article["title"] + ". " + (article.get("description") or "")
        score = cosine_similarity(question_vec, get_embedding(text))
        scored.append((score, article))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [a["title"] for _, a in scored[:top_n]]


def build_crew(ticker, question):
    stock_summary = get_stock_summary(ticker)
    top_headlines = get_top_news(question, ticker)
    headlines_text = "\n".join([f"- {h}" for h in top_headlines])

    market_task = Task(
        description=f"Here is the current market data for {ticker}:\n{stock_summary}\n"
                     f"Summarize this in 1-2 sentences, sticking strictly to these facts.",
        expected_output="A short factual summary of the stock's current price and movement.",
        agent=market_analyst,
    )

    news_task = Task(
        description=f"Here are recent news headlines about {ticker}:\n{headlines_text}\n"
                     f"Summarize what these headlines suggest about the stock, in 2-3 sentences.",
        expected_output="A concise summary of news sentiment and relevance.",
        agent=news_analyst,
    )

    advisor_task = Task(
        description=f"The user asked: '{question}'\n"
                     f"Using the market summary and news summary from your teammates, "
                     f"write a clear final answer (3-4 sentences). Include a brief disclaimer "
                     f"that this isn't financial advice.",
        expected_output="A final, well-rounded answer for the user.",
        agent=advisor,
        context=[market_task, news_task],
    )

    return Crew(
        agents=[market_analyst, news_analyst, advisor],
        tasks=[market_task, news_task, advisor_task],
        process=Process.sequential,
    )


if __name__ == "__main__":
    crew = build_crew("AAPL", "Why did this stock move this week?")
    result = crew.kickoff()
    print("\n--- FINAL ANSWER ---")
    print(result)