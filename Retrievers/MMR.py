from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

#step1:your source documents
documents=[
    Document(page_content="""LangChain is a framework for developing applications powered by large language models.
It provides tools and abstractions for building applications such as chatbots,
question-answering systems, retrieval-augmented generation (RAG), and AI agents.
LangChain supports prompt templates, language models, output parsers, retrievers,
document loaders, and chains that help developers connect different components
together."""),
    Document(page_content="""LangChain provides document loaders and retrievers that allow applications to
work with external information. Document loaders can load data from sources such
as PDFs, text files, websites, and databases. Retrievers are used to search for
relevant documents based on a user's query. These components are commonly used
in Retrieval-Augmented Generation (RAG) applications."""),
    Document(page_content="""LangGraph can be used to build sophisticated AI agents and multi-agent systems.
Agents can maintain state, call tools, make decisions, and execute tasks through
different nodes in a graph. Conditional edges allow the workflow to dynamically
choose the next step based on the current state. This makes LangGraph useful
for building reliable and controllable agentic applications."""),
    Document(
    page_content="""Chroma is a vector database used in AI and RAG applications.
It stores document embeddings and supports similarity search. Chroma can be
integrated with LangChain to store embeddings and retrieve documents that are
semantically relevant to a user's query.""")
]

#step2:Initialize embedding model 
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#step3:Create schema vector store in memory
vector_store=FAISS.from_documents(
    documents=documents,
    embedding=embedding_model,
)

#step 4:convert vector store into retriever
retriever=vector_store.as_retriever(search_type="mmr",search_kwargs={"k":2,"lambda_mult":1})

query="what is langchain?"
results=retriever.invoke(query)

for i,doc in enumerate(results):
    print(f"\n---result{i+1}---")
    print(doc.page_content)