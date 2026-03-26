from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent      
from langchain_core.messages import HumanMessage

import os
from dotenv import load_dotenv
load_dotenv()

# -- Model ------------------------------------------------------------------
MODEL_NAME  = os.getenv("model", "qwen3:5b")
TEMPERATURE = float(os.getenv("temperature", "0.7"))

llm = ChatOllama(
    model=MODEL_NAME, 
    temperature=TEMPERATURE  
)

# -- Tools ────----------------------------------------------------------------

from langchain_community.utilities import (
    WikipediaAPIWrapper, 
    DuckDuckGoSearchAPIWrapper, 
    ArxivAPIWrapper
)
from langchain_community.tools import (
    WikipediaQueryRun, 
    DuckDuckGoSearchResults, 
    ArxivQueryRun
)

ddgs_api    = DuckDuckGoSearchAPIWrapper(max_results=3)
ddgs_search = DuckDuckGoSearchResults(
    api_wrapper=ddgs_api, 
    name="web_search_ddgs"
)

wiki_api= WikipediaAPIWrapper(doc_content_chars_max=200, top_k_results=3)
wiki_data = WikipediaQueryRun(
    name="Wikipedia", 
    api_wrapper=wiki_api
)

arxiv_api= ArxivAPIWrapper(doc_content_chars_max=200, top_k_results=3)
arxiv_data  = ArxivQueryRun(
    name="Arxiv", 
    api_wrapper=arxiv_api
)

#---Tools Functions----------------------------------------------------------------
def run_tool(tool_name: str, query: str, run_fn) -> str:
    if input(f"\nAllow [{tool_name}]? [y/n]: ").strip().lower() == "y":
        return run_fn(query)
    print("\n---- Tool Call Rejected! ----")
    return "Tool Call Rejected"

def ddgs_fun(query: str) -> str:
    """Search the web using DuckDuckGo."""
    return run_tool("DDGS Web Search", query, ddgs_search.run)

def wiki_fun(query: str) -> str:
    """Search Wikipedia for information."""
    return run_tool("Wikipedia Search", query, wiki_data.run)

def arxiv_fun(query: str) -> str:
    """Search Arxiv for research papers."""
    return run_tool("Arxiv Search", query, arxiv_data.run)

# --- SQLite Mem----------------------------------------------------------------

from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

DB_PATH = "chat_history.db"
conn    = sqlite3.connect(DB_PATH, check_same_thread=False)
memory  = SqliteSaver(conn)

# --- Agent -------------------------------------------------------------------

agent = create_react_agent(
    model=llm,
    tools=[wiki_fun, arxiv_fun, ddgs_fun],
    prompt="You are a Helpful AI Assistant",
    checkpointer=memory,
)

memory.setup()

# --- Session Masseges----------------------------------------------------------------

import uuid

def pick_session() -> str:
    rows = conn.execute(
        "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
    ).fetchall()

    if rows:
        print("\nExisting sessions:")
        for i, (tid,) in enumerate(rows, 1):
            print(f"  [{i}] {tid}")
        print("  [N] Start a new session")
        choice = input("Pick a session number or N: ").strip()
        if choice.lower() != "n":
            try:
                return rows[int(choice) - 1][0]
            except (ValueError, IndexError):
                print("Invalid choice — starting a new session.")

    new_id = str(uuid.uuid4())
    print(f"\nNew session: {new_id}")
    return new_id

# --- Main Loop ----------------------------------------------------------------
thread_id = pick_session()
config    = {"configurable": {"thread_id": thread_id}}

print("\nEnter a question, or 'q' / 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() in ("q", "quit"):
        print("Goodbye!")
        conn.close()
        break

    # Hot-swap temperature mid-session
    if user_input.startswith("/temp "):
        try:
            llm.temperature = float(user_input.split()[1])
            print(f"Temperature set to {llm.temperature}")
        except (IndexError, ValueError):
            print("Usage: /temp 0.9")
        continue

    print("Agent: ", end="", flush=True)
    final_text = ""

    for chunk in agent.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="values",
    ):
        last_msg = chunk["messages"][-1]
        if last_msg.type == "ai" and last_msg.content:
            new_text = last_msg.content
            if new_text.startswith(final_text):
                print(new_text[len(final_text):], end="", flush=True)
                final_text = new_text

    print()