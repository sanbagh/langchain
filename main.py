from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()
class Source(BaseModel):
    """Schema for a source used by agent"""
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer:str = Field(description="The agent's answer to the query")
    sources:List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

#tavily = TavilyClient()

# @tool
# def search(query:str) ->str:
#     """
#     Tool that seraches over internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search result
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query=query)

llm = ChatOllama(model="llama3.2")
#llm = ChatGroq(model="openai/gpt-oss-120b")
# tools =[search]
# tools = [TavilySearch(max_results=5,
#         topic="general",
#         search_depth="basic",
#         include_answer=True)]

tools = [TavilySearch(topic="general",time_range="week")]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main() -> None:
    # result = agent.invoke({"messages": HumanMessage(content="What is the current weather in Delhi, India? "
    #                     "Search for today's weather and cite your sources?")})
    # print(result)
    result = agent.invoke({"messages": HumanMessage(content="Search for 3 job posting for an ai engineer using langchain in India on linkedin and list their details?")})
    print(result)
    

if __name__ == "__main__":
    main();