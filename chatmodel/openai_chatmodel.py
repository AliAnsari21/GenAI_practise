from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
model=ChatOpenAI(model='gpt-4',temperature=0.5)
result=model.invoke("what is capital of india")
print(result)#this give content,metadata,args and source everything
print(result.content)#this only gives content