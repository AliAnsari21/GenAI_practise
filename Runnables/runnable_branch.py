from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch,
    RunnableSequence
)
from dotenv import load_dotenv
load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)
parser=StrOutputParser()

prompt1=PromptTemplate(
    template='write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='summarize the following text {text}',
    input_variables=['text']
)


report_gen_chain=RunnableSequence(prompt1,model,parser)

branch_chain=RunnableBranch(
    (lambda x:len(x.split())>500,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain=RunnableSequence(report_gen_chain,branch_chain)

print(final_chain.invoke({'topic':'Russia vs Ukraine'}))