
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatOllama(model="llama3.2")

messages = [
    SystemMessage(content="You are a helpful teacher. Explain in simple Hindi."),
    HumanMessage(content="What is C#?")
]

response = model.invoke(messages)

print(response.content)
