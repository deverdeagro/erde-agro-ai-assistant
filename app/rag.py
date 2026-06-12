from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    TextLoader
)

from pathlib import Path


SOURCE_TYPE_MAP = {
    "01_company_profile": "company_profile",
    "02_products_and_services": "product",
    "03_faq": "faq",
    "04_sales_questions": "sales",
}


def get_source_type(filename: str) -> str:
    for key, value in SOURCE_TYPE_MAP.items():
        if key in filename:
            return value
    return "general"


def load_documents():

    documents = []
    data_path = Path("data")

    # PDFs
    pdf_loader = PyPDFDirectoryLoader(str(data_path))
    pdf_docs = pdf_loader.load()
    for doc in pdf_docs:
        doc.metadata["source_type"] = get_source_type(doc.metadata.get("source", ""))
    documents.extend(pdf_docs)

    # TXT
    for file in data_path.glob("*.txt"):
        loader = TextLoader(str(file), encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_type"] = get_source_type(str(file))
        documents.extend(docs)

    clean_docs = [doc for doc in documents if doc.page_content.strip()]

    print(f"Loaded {len(clean_docs)} valid documents")

    return clean_docs