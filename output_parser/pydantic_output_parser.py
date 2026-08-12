from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)

class person(BaseModel):
    name:str=Field(description='Name of person')
    age:int=Field(gt=18,description='age of person')
    city:str=Field(description='city name where the person belongs to')

parser=PydanticOutputParser(pydantic_object=person)

template=PromptTemplate(
    template='generate the name ,age and city of a fictiona {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)
chain=template|model|parser
result=chain.invoke({'place':'srilanka'})
print(result)