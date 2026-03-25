from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

# ========================================================
import os
from dotenv import load_dotenv
load_dotenv()

MODEL_NAME = os.getenv("model","qwen3.5:397b-cloud")
TEMPERATURE = float(os.getenv("temperature","0.7"))

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE
)
#===========================================================

from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchResults, ArxivQueryRun

#===========================================================

ddgs_api = DuckDuckGoSearchAPIWrapper(max_results=3)
ddgs_search = DuckDuckGoSearchResults(
    api_wrapper=ddgs_api,
    name="web_search_ddgs",
    description="You are a ddgs web assistent"
)

wiki_api = WikipediaAPIWrapper(doc_content_chars_max=200,top_k_results=3)
wiki_data = WikipediaQueryRun(
    name="Wikipedia",
    api_wrapper=wiki_api,
    description="You are a helpful wikipedia Searcher"
)

arxiv_api = ArxivAPIWrapper(doc_content_chars_max=200,top_k_results=3)
arxiv_data = ArxivQueryRun(
    name="Arxiv",
    api_wrapper=arxiv_api,
    description="You are a helpful Arxiv Searcher"
)
#==============================================================

def run_tool(tool_n:str,query:str,run_fun)->str:
    # print(f"Tool Name = {tool_n}")
    # print(f"Query to run = {query}")
    if input(f"\nEnetr [y/n] for permission for {tool_n} = ") == "y":
        return run_fun(query)
    else:
        print("\n----Tool Call Rejected!-----")
        return "Tool Call Rejected"

def ddgs_fun(query,str)->str:
    """It is a DDGS Web Search Function"""
    return run_tool("DDGS Web Search",query,ddgs_search.run)

def wiki_fun(query,str)->str:
    """It is a Wikipedia Search"""
    return run_tool("Wikipedia Search",query,wiki_data.run)

def arxiv_fun(query,str)->str:
    """It is a Arxiv Search"""
    return run_tool("Arxiv Search",query,arxiv_data.run)

#=============================================================

agent = create_react_agent(
    model=llm,
    tools=[wiki_fun,arxiv_fun,ddgs_fun],
    prompt="You are a Helpfull AI Assistent"
)

print("Enter Question to get Answer or q or quit for end conversation\n")
while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue
    elif user_input.lower() in ("q","quit"):
        print("Goodby!")
        break
    else:
        response = agent.invoke(
           {"messages":[{"role":"user","content": user_input }]}
           )
        print(response["messages"][-1].content)


