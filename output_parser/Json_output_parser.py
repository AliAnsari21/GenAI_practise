from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)
parser=JsonOutputParser()

template=PromptTemplate(
    template='give me age ,name and city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)
chain=template|model|parser
result=chain.invoke({})
print(result)