from langchain_ollama import OllamaLLM


llm = OllamaLLM(
    model="llama3.1:8b",
    base_url="http://localhost:11434"
)


def ask(question):
    return llm.invoke(question)