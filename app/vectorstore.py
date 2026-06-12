from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import load_documents


def _get_embeddings():
    return OllamaEmbeddings(
        model="mxbai-embed-large",
        base_url="http://localhost:11434"
    )


def create_vectorstore():
    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "],
    )
    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=_get_embeddings(),
        persist_directory="./chroma_db"
    )
    return vectorstore


def load_vectorstore():
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=_get_embeddings()
    )
