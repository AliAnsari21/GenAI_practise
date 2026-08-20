from langchain_core.tools import InjectedToolArg,tool
from typing import Annotated
from langchain_groq import ChatGroq
import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()
@tool 
def get_conversion_factor(base_currency:str,target_currency:str)->float:
    """this function fetches the currency conversion factor between base currency and target currency"""

    url=f"https://v6.exchangerate-api.com/v6/da631676238cbc8fbd47b1f3/latest/USD"
    response=requests.get(url)
    return response.json()["conversion_rates"][target_currency]

@tool
def convert(base_currency_value:int,conversion_rate:Annotated[float,InjectedToolArg])->float:
    """give a currency conversion rate,this function calculate the target currency value from a given base currency value"""
    return base_currency_value*conversion_rate


get_conversion_factor.invoke({'base_currency':'USD','target_currency':'INR'})
convert.invoke({'base_currency_value':10,'conversion_rate':95.7489})

#tool binding
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
llm_with_tool = llm.bind_tools([get_conversion_factor,convert])

messages=[HumanMessage('what is the conversion factor between usd and inr based on that can you convert 10 usd to inr')]

ai_message=llm_with_tool.invoke(messages)
messages.append(ai_message)
ai_message.tool_calls
import json
for tool_call in ai_message.tool_calls:
    #execute the first tool and get values
    if tool_call['name']=='get_conversion_factor':
        tool_message1=get_conversion_factor.invoke(tool_call)
        #fetch this conversion rate
        conversion_rate = float(tool_message1.content)
        #append this tool message to message list
        messages.append(tool_message1)

    #execute the 2nd tool using conversion rate from tool 1
    if tool_call['name']=='convert':
        #fetch the current arg
        tool_call['args']['conversion_rate']=conversion_rate
        tool_message2=convert.invoke(tool_call)
        messages.append(tool_message2)

llm_with_tool.invoke(messages).content