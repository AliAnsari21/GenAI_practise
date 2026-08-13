from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)
prompt=PromptTemplate(
    template='suggest a catchy blog title about {topic}',
    input_variables=['topic']
)
topic=input("enter a topic")
formatted_prompt=prompt.format(topic=topic)

blog_title=model.invoke(formatted_prompt)
print("generate blog title",blog_title)