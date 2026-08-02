import ollama

text = "Apple stock dropped today"

response = ollama.embeddings(model="nomic-embed-text", prompt=text)

vector = response["embedding"]

print("Vector length:", len(vector))
print("First 5 numbers:", vector[:5])