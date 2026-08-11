from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)

messages=[
    SystemMessage(content='you are a helpful assistant'),
    HumanMessage(content='tell me about lagchain')
]
result=model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)