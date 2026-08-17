from langchain_core.tools import tool
#step1 create function
def multiply(a,b):
    """Multiply two numbers"""
    return a*b

#step2 add type hints
def multiply(a:int ,b:int)->int:
    """Multiply two numbers"""
    return a*b

#step3 add tool decorator
@tool
def multiply(a:int ,b:int)->int:
    """Multiply two numbers"""
    return a*b

result=multiply.invoke({"a":3,"b":5})
print(result)