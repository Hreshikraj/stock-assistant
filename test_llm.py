import ollama

response = ollama.chat(
    model="llama3.1",
    messages=[
        {"role": "user", "content": "What is 2 + 2? Answer in one short sentence."}
    ]
)

print(response["message"]["content"])