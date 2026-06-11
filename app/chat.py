from vectorstore import create_vectorstore
from llm import ask


db = create_vectorstore()


retriever = db.as_retriever(
    search_kwargs={
        "k":3
    }
)


while True:

    question = input(
        "\nYou: "
    )


    if question.lower()=="exit":
        break


    docs = retriever.invoke(
        question
    )


    context = "\n\n".join(
        [
            d.page_content
            for d in docs
        ]
    )


    prompt = f"""
You are ERDE Agro's AI assistant.

Answer only using the context.

Context:

{context}


Question:

{question}

"""


    answer = ask(prompt)


    print(
        "\nAI:",
        answer
    )