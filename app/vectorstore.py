from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import load_documents


def create_vectorstore():

    documents = load_documents()


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )


    chunks = splitter.split_documents(
        documents
    )


    print(
        f"Created {len(chunks)} chunks"
    )


    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )


    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )


    return vectorstore