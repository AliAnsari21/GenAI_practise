from pydantic import BaseModel,Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal

load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)
class review(BaseModel):
    key_themes:list[str]=Field(description="write down all the themes discussed in the reviewin a list")
    summary:str=Field(description="a brief summary of review")
    sentiment:Literal["pos","neg"]=Field(description="Return sentiment of review")
    pros:Optional[list[str]]=Field(description="write down all pros inside a list")
    cons:Optional[list[str]]=Field(description="write down all cons inside a list")
    name:Optional[str]=Field(default=None,description="name of author")

structured_model=model.with_structured_output(review)
result=structured_model.invoke(    """
    My name is Rahul. I recently bought the Sony WH-1000XM5 headphones
    and I am very impressed with the sound quality and noise cancellation.
    The battery life is excellent and the headphones are comfortable to
    wear for long hours. However, they are quite expensive and the case
    could be better. Overall, I am very happy with my purchase.
    """
)
print(result.name)
    