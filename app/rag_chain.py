import operator
from typing import Annotated, List, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from sentence_transformers import CrossEncoder

from llm import llm
from vectorstore import load_vectorstore

GREETING_KEYWORDS = {
    "hi", "hello", "hey", "thanks", "thank you",
    "bye", "goodbye", "good morning", "good afternoon", "good evening",
}

# Cross-encoder scores below this mean the question has no relevant match in our docs
RELEVANCE_THRESHOLD = 0.0


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    question: str
    rewritten_query: str
    retrieved_docs: List[Document]
    answer: str
    is_greeting: bool


def create_rag_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8},
    )

    # Downloads ~100MB on first run, cached after that
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def detect_greeting(state: AgentState) -> dict:
        q = state["question"].lower().strip()
        words = set(q.split())
        is_greeting = bool(words & GREETING_KEYWORDS) and len(q.split()) <= 6
        return {"is_greeting": is_greeting}

    def rewrite_query(state: AgentState) -> dict:
        if state.get("is_greeting"):
            return {"rewritten_query": ""}

        history = state.get("messages", [])
        if not history:
            return {"rewritten_query": state["question"]}

        history_text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in history[-4:]
        )
        prompt = (
            f"Conversation history:\n{history_text}\n\n"
            f"Rewrite this as a standalone search query capturing full context:\n"
            f"Question: {state['question']}\n"
            f"Standalone query:"
        )
        response = llm.invoke(prompt)
        rewritten = response.content.strip() if hasattr(response, "content") else str(response).strip()
        return {"rewritten_query": rewritten}

    def retrieve_and_rerank(state: AgentState) -> dict:
        if state.get("is_greeting"):
            return {"retrieved_docs": []}

        docs = retriever.invoke(state["rewritten_query"])
        if not docs:
            return {"retrieved_docs": []}

        pairs = [(state["question"], doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)

        if max(scores) < RELEVANCE_THRESHOLD:
            return {"retrieved_docs": []}

        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return {"retrieved_docs": [doc for _, doc in ranked[:4]]}

    def generate_answer(state: AgentState) -> dict:
        if state.get("is_greeting"):
            answer = (
                "Hello! I'm the ERDE Agro AI Assistant. "
                "I can help you with information about our products, services, and farming solutions. "
                "What would you like to know?"
            )
            return {
                "answer": answer,
                "messages": [HumanMessage(content=state["question"]), AIMessage(content=answer)],
            }

        docs = state.get("retrieved_docs", [])
        if not docs:
            answer = (
                "I can only answer questions about ERDE Agro products, services, and farming solutions. "
                "That question is outside the scope of my knowledge base. "
                "Please contact ERDE Agro directly if you need further assistance."
            )
            return {
                "answer": answer,
                "messages": [HumanMessage(content=state["question"]), AIMessage(content=answer)],
            }

        context = "\n\n".join(doc.page_content for doc in docs)
        history = state.get("messages", [])[-6:]

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are the ERDE Agro AI Assistant. Answer using ONLY the provided context.\n\n"
                "Rules:\n"
                "- Be clear, helpful, and concise\n"
                "- Do not add information not in the context\n"
                "- If the context doesn't contain the answer, say: "
                "\"I don't have that information in my company documents.\"\n"
                "- Use plain text, no markdown\n\n"
                "Context:\n{context}",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        messages = prompt.format_messages(
            context=context,
            history=history,
            question=state["question"],
        )
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        return {
            "answer": answer,
            "messages": [HumanMessage(content=state["question"]), AIMessage(content=answer)],
        }

    graph = StateGraph(AgentState)
    graph.add_node("detect_greeting", detect_greeting)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve_and_rerank", retrieve_and_rerank)
    graph.add_node("generate_answer", generate_answer)

    graph.set_entry_point("detect_greeting")
    graph.add_edge("detect_greeting", "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve_and_rerank")
    graph.add_edge("retrieve_and_rerank", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile(checkpointer=MemorySaver())
