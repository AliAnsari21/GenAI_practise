from langchain_groq import ChatGroq

from langchain_core.tools import tool

import requests

from langchain_community.tools import DuckDuckGoSearchRun

from dotenv import load_dotenv

load_dotenv()

search_tool=DuckDuckGoSearchRun()

results=search_tool.invoke('top news in india today')

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

from langchain.agents import create_agent

# step 3 create the agent

agent=create_agent(
    model=llm,
    tools=[search_tool]
)

# step 5 invoke

response=agent.invoke({
    "messages":[
        {
            "role":"user",
            "content":"3 ways to reach goa from delhi"
        }
    ]
})

print(response["messages"][-1].content)