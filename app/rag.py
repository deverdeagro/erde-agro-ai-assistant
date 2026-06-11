from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    TextLoader
)

from pathlib import Path


def load_documents():

    documents = []

    data_path = Path("data")


    # PDFs
    pdf_loader = PyPDFDirectoryLoader(
        str(data_path)
    )

    pdf_docs = pdf_loader.load()

    documents.extend(pdf_docs)


    # TXT
    for file in data_path.glob("*.txt"):

        loader = TextLoader(
            str(file),
            encoding="utf-8"
        )

        documents.extend(
            loader.load()
        )


    # remove empty docs
    clean_docs = []

    for doc in documents:

        if doc.page_content.strip():
            clean_docs.append(doc)


    print(
        f"Loaded {len(clean_docs)} valid documents"
    )


    return clean_docs