from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)

prompt=PromptTemplate(
    template='generate 5 intresting fact about {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()

chain=prompt|model|parser
result=chain.invoke({'topic':'blackhole'})
print(result)

chain.get_graph().print_ascii()