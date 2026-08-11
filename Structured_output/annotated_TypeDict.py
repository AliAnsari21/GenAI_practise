from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal

load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)

class review(TypedDict):
    summary:Annotated [str,"a brief summary of review"]
    sentiment:Annotated [str,"Return sentiment of review"]
    key_themes:Annotated [list[str],"write down all the themes discussed in the reviewin a list"]
    pros:Annotated[Optional[list[str]],"write down all pros inside a list"]
    cons:Annotated[Optional[list[str]],"write down all cons inside a list"]
    sentiment:Annotated [Literal["pos","neg"],"Return sentiment of review"]

structured_model=model.with_structured_output(review)
result=structured_model.invoke( "The product quality is excellent and delivery was very fast. "
    "I am very happy with my purchase.")
print(result)