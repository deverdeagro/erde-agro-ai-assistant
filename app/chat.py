from rag_chain import create_rag_chain

chain = create_rag_chain()
session_id = "default_session"

print("ERDE Agro AI Assistant ready. Type 'exit' to quit.\n")

while True:
    question = input("You: ").strip()

    if not question:
        continue

    if question.lower() == "exit":
        print("Goodbye!")
        break

    result = chain.invoke(
        {"question": question},
        config={"configurable": {"thread_id": session_id}},
    )

    print(f"\nAI: {result['answer']}\n")