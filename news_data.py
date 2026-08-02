import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("NEWSAPI_KEY")

params = {
    "q": "Apple stock",
    "sortBy": "publishedAt",
    "language": "en",
    "pageSize": 5,
    "apiKey": api_key,
}

response = requests.get("https://newsapi.org/v2/everything", params=params)
data = response.json()

for article in data["articles"]:
    print(article["title"])
    print(article["publishedAt"])
    print("---")