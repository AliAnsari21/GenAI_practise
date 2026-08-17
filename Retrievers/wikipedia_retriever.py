from langchain_community.retrievers import WikipediaRetriever

#initialize the retriever 
retriever=WikipediaRetriever(top_k_results=2,lang='en')

#define your query
query="the geopolitical history of India and Pakistan from the perspective of a chinese"

#get relevant wikipedia document
docs=retriever.invoke(query)

#print retrieve content
for i,doc in enumerate(docs):
    print(f"\n---result{i+1}---")
    print(f"content:\n{doc.page_content}")