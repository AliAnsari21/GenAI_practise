from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

#load the document
loader=TextLoader("docs.txt")#ensure doc text exists
documents=loader.load()

#split text into smaller chunk
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
docs=text_splitter.split_documents(documents)

#convert text into embeddings and store in Faiss
vector_store=FAISS.from_documents(docs,HuggingFaceEmbeddings())

#create a retriever(fetch relevant documents)
retriever=vector_store.as_retriever()

#manually retrieve relevant documents
query="what are the key takeaways from the document?"
retrieved_docs=retriever._get_relevant_documents(query)

#combined retrieved text into a simple prompt
retrieved_text="\n".join([doc.page_content for doc in retrieved_docs])

#initialize llm
model=Groq(model="llama-3.3-70b-versatile",temperature=0)

#manually passed retrieved text to llm
prompt=f"Based on the following text answer the question: {query}\n\n{retrieved_text}"
answer=model.invoke(prompt)
print(answer)