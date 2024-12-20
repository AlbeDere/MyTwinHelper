import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])
vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-4o-mini")

qa = RetrievalQA.from_chain_type(
    llm=chat, chain_type="stuff", retriever=vectorstore.as_retriever()
)    

res = qa.invoke("Tell me more about your professional experience.")
print(res) 
print("\n")
res = qa.invoke("What is your educational background?")
print(res)
print("\n")

res = qa.invoke("What are your skills?")
print(res)
print("\n")

res = qa.invoke("Contact information?")
print(res)
print("\n")

res = qa.invoke("Who are you?")
print(res)
print("\n")

res = qa.invoke("What is your prefered techstack?")
print(res)
print("\n")

res = qa.invoke("Professional philosophy?")
print(res)