from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
    RunnableSequence,
    RunnableBranch
)
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)
parser1=StrOutputParser()

class Feedback(BaseModel):
    sentiment:Literal['positive','negative']=Field(description='Give the sentiment of feedback')

parser2=PydanticOutputParser(pydantic_object=Feedback)

prompt1=PromptTemplate(
    template='classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)
classifier_chain=prompt1|model|parser2

prompt2=PromptTemplate(
    template='write an appropriate response to this positive\n {feedback}',
    input_variables=['feedback'],
    
)

prompt3=PromptTemplate(
    template='write an appropriate response to this negative\n {feedback}',
    input_variables=['feedback'],   
)

branch_chain=RunnableBranch(
    (lambda x:x.sentiment=='positiive',prompt2|model|parser1),
    (lambda x:x.sentiment=='negative',prompt3|model|parser1),
    RunnableLambda(lambda x:"could not find sentiment")
)

chain=classifier_chain|branch_chain
result=chain.invoke({'feedback':'this is terrible phone'})
print(result)