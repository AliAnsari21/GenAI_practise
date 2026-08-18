from langchain_core.tools import BaseTool
from pydantic import BaseModel,Field
from typing import Type


#arg schema using pydantic
class MultiplyInput(BaseModel):
    a:int=Field(required=True,description="the first number to be add")
    b:int=Field(required=True,description="the second number to be add")

class MultiplyTool(BaseTool):
    name:str="multiply"
    description:str="multiply 2 number"
    args_schema:Type[BaseModel]=MultiplyInput

    def _run(self,a:int,b:int)->int:
        return a*b

multiply_tool=MultiplyTool()
result=multiply_tool.invoke({"a":3,"b":5})
print(result)