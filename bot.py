import os
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


embeddings = OpenAIEmbeddings(openai_api_type=os.environ.get("OPENAI_API_KEY"))
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-4o-mini")

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm=llm, retriever=retriever, prompt=contextualize_q_prompt
)

qa_system_prompt = (
    "You are a virtual assistant designed to answer questions specifically about Albert Derevski. "
    "Use the following pieces of retrieved context to answer the question, ensuring the response is relevant to Albert's "
    "professional experience, skills, education, or portfolio. If you don't know the answer or the question is unrelated to Albert, "
    "respond politely with, 'I'm sorry, I can only answer questions about Albert Derevski.' Avoid speculation or fabricated information. "
    "Keep the response concise, professional.\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain
)

chat_history = []  

print("Welcome! Ask me questions about Albert Derevski. Type 'exit' to quit.")

while True:
    query = input("You: ")
    if query.lower() == "exit":
        print("Goodbye!")
        break

    response = rag_chain.invoke({"input": query, "chat_history": chat_history})
    print(f"Assistant: {response['answer']}")
    
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response['answer']))
