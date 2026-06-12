from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="qwen2.5:14b",
    base_url="http://localhost:11434",
)


def ask(question: str) -> str:
    return llm.invoke(question).content