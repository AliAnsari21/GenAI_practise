from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
    RunnableSequence
)
from dotenv import load_dotenv
load_dotenv()
prompt=PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)

model=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)
parser=StrOutputParser()
chain=prompt|model|parser
print(chain.invoke({'topic':'AI'}))