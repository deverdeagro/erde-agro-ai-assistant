from rag import load_documents


docs = load_documents()


print("Total documents:", len(docs))


for i, doc in enumerate(docs):

    print("\n===================")
    print("DOCUMENT:", i)

    print("SOURCE:")
    print(doc.metadata)

    content = doc.page_content

    print("CONTENT LENGTH:")
    print(len(content))

    print("CONTENT PREVIEW:")
    print(content[:300])