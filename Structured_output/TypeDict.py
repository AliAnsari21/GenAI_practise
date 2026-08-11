from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)

class review(TypedDict):
    summary:str
    sentiment:str

structured_model=model.with_structured_output(review)
result=structured_model.invoke( "The product quality is excellent and delivery was very fast. "
    "I am very happy with my purchase.")
print(result)