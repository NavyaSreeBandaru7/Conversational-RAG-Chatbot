from __future__ import annotations

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from .config import get_settings, require_openai_key
from .utils import format_docs_for_context, format_sources

def build_chat():
    load_dotenv()
    settings = get_settings()
    require_openai_key()

    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    vs = FAISS.load_local(
        str(settings.index_dir),
        embeddings,
        allow_dangerous_deserialization=True,  # required by FAISS load_local
    )

    retriever = vs.as_retriever(search_kwargs={"k": settings.k})

    llm = ChatOpenAI(model=settings.chat_model, temperature=settings.temperature)

    # 1) Rewrite follow-up questions into standalone questions (improves retrieval)
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the user's latest question into a standalone question using chat history. Return ONLY the question."),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    # 2) Answer grounded in retrieved context
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Use the provided context to answer. If the context is insufficient, say what you can conclude and what you cannot. "
         "Do NOT invent details. Prefer concise, clear answers."),
        MessagesPlaceholder("history"),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
    ])

    rag_chain = (
        RunnablePassthrough.assign(question=rewrite_chain)
        .assign(docs=(lambda x: x["question"]) | retriever)
        .assign(context=(lambda x: x["docs"]) | RunnableLambda(format_docs_for_context))
        .assign(output=answer_prompt | llm | StrOutputParser())
    )

    # 3) Memory for multi-turn chat
    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    chat = RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    def ask(q: str, session_id: str = "default"):
        out = chat.invoke({"input": q}, config={"configurable": {"session_id": session_id}})
        return out["output"], format_sources(out["docs"])

    return ask
