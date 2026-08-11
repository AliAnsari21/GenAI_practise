from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
embedding=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
text="Delhi is capital of India"
result=embedding.embed_query(text)#for emedding query
print(str(result))