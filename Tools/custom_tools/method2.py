from langchain_core.tools import BaseTool
from pydantic import BaseModel,Field
from typing import Type
from langchain_core.tools import StructuredTool

#arg schema using pydantic
class MultiplyInput(BaseModel):
    a:int=Field(required=True,description="the first number to be add")
    b:int=Field(required=True,description="the second number to be add")


def multiply_func(a:int,b:int)->int:
    return a*b

multiply_tool=StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="multiply 2 number",
    args_schema=MultiplyInput
)
result=multiply_tool.invoke({"a":3,"b":5})
print(result)