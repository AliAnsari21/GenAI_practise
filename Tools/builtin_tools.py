from langchain_community.tools import DuckDuckGoSearchRun
search=DuckDuckGoSearchRun()
result=search.invoke('sri-lanka vs india news')
print(result)

"""
some more in built tools
1 WikipediaQueryRun
2 PythonREPLTool
3 ShellTool
4 RequestsGetTool
5 GmailSendMessageTool
and many more
"""