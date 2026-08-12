from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)

schema=[
    ResponseSchema(name='fact-1',description='fact 1 about topic'),
    ResponseSchema(name='fact-2',description='fact 2 about topic'),
    ResponseSchema(name='fact-3',description='fact 3 about topic')
]
parser=StructuredOutputParser.from_response_schema(schema)

template=PromptTemplate(
    template='give 3 facts about the {topic}\n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain=template|model|parser
result=chain.invoke({'topic':'blackhole'})
print(result)