from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)
chat_template=ChatPromptTemplate([
    ('system','you are helpful {domain} expert'),
    ('human','expalin in simple term ,what is {topic}')
])
prompt=chat_template.invoke({'domain':'cricket','topic':'long off'})
result=model.invoke(prompt)
chat_template.append(AIMessage(content=result.content))
print(chat_template)
