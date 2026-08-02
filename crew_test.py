import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.llm import LLM

load_dotenv()

llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
)

analyst = Agent(
    role="Market Analyst",
    goal="Explain stock market concepts clearly",
    backstory="You are an experienced financial analyst who explains things simply.",
    llm=llm,
)

task = Task(
    description="Explain what a stock's 'previous close' means, in 2 sentences.",
    expected_output="A short, clear explanation.",
    agent=analyst,
)

crew = Crew(agents=[analyst], tasks=[task])

result = crew.kickoff()
print(result)