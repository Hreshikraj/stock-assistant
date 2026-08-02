import ollama
import numpy as np

def get_embedding(text):
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

question = "Why did Apple stock move this week?"

article_1 = "Apple's $5tn moment shows investors are questioning the AI race"
article_2 = "Xiaomi Ecosystem: It just works, too"

q_vec = get_embedding(question)
a1_vec = get_embedding(article_1)
a2_vec = get_embedding(article_2)

print("Similarity to article 1 (Apple):", cosine_similarity(q_vec, a1_vec))
print("Similarity to article 2 (Xiaomi):", cosine_similarity(q_vec, a2_vec))