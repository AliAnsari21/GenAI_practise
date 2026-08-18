from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# tool creation

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

print(multiply.invoke({"a": 3, "b": 5}))

# tool binding
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
llm_with_tool = llm.bind_tools([multiply])


# tool calling
query = HumanMessage('can you multiply 3 with 10')
messages = [query]
result = llm_with_tool.invoke('hii how are you')
messages.append(result)
result = llm_with_tool.invoke('can you multiply 3 with 10')
print(result.tool_calls[0]['args'])

# tool execution
multiply.invoke(result.tool_calls[0]['args'])
tool_result = multiply.invoke(result.tool_calls[0]['args'])
messages.append(tool_result)
llm_with_tool.invoke(messages)
print(messages)