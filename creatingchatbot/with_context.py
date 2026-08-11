from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)


chat_history=[SystemMessage(content="you are helpful assistant")]
while True:
    user_input=input('you:')
    chat_history.append(HumanMessage(content=user_input))
    if user_input=='exit':
        break
    result=model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI:",result.content)

print(chat_history)