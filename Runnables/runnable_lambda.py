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

model=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)
parser=StrOutputParser()

prompt1=PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)

joke_gen_chain=RunnableSequence(prompt1,model,parser)

def word_count(text):
    return len(text.split())

parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':RunnableLambda(word_count)
})

final_chain=RunnableSequence(joke_gen_chain,parallel_chain)

print(final_chain.invoke({'topic':'cricket'}))