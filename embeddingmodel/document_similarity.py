from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
load_dotenv()
embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
documents=[
    "Virat Kohli is a famous Indian cricketer known for his batting skills.",
    "Rohit Sharma is an Indian cricketer and an excellent opening batsman.",
    "MS Dhoni is a former Indian captain known for his leadership and wicketkeeping.",
    "Jasprit Bumrah is an Indian fast bowler famous for his unique bowling action.",
    "Babar Azam is a Pakistani cricketer known for his elegant batting style."
]
query="tell me about Virat Kohli"
docs_embedding=embedding.embed_documents(documents)
query_embedding=embedding.embed_query(query)
print(cosine_similarity([query_embedding],docs_embedding)[0])
scores=cosine_similarity([query_embedding],docs_embedding)[0]
print(sorted(list(enumerate(scores)),key=lambda x:x[1])[-1])
index,scores=sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]
print(query)
print(documents[index])
print("similarity score is:",scores)