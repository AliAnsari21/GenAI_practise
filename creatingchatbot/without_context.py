from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt
load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)

while True:
    user_input=input('you:')
    if user_input=='exit':
        break

    result=model.invoke(user_input)
    print("AI:",result.content)