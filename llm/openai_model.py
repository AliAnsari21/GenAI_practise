from langchain_openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
llm=OpenAI(model="gpt-3.5-turbo-instruct")
result=llm.invoke("what is capital of india")
print(result)#this give content,metadata,args and source everything
print(result.content)#this only gives content