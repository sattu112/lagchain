import os
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.chat_models import ChatOpenAI
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- Tool 1: Web Search (Tavily) ---
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(query):
    response = tavily.search(query=query, max_results=5)
    results = [r["content"] for r in response["results"]]
    return "\n".join(results)

web_tool = Tool(
    name="Web Search",
    func=web_search,
    description="Useful for searching recent and detailed information from the internet."
)

# --- Tool 2: Wikipedia ---
wiki = WikipediaAPIWrapper()
wiki_tool = WikipediaQueryRun(api_wrapper=wiki)

# --- Agent ---
tools = [web_tool, wiki_tool]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# --- Report Generator ---
def generate_report(topic):
    prompt = f"""
    Research the topic: {topic}
    
    Then generate a structured report with:
    - Cover Page
    - Title
    - Introduction
    - Key Findings
    - Challenges
    - Future Scope
    - Conclusion
    """

    return agent.run(prompt)


if __name__ == "__main__":
    topic = input("Enter topic: ")
    report = generate_report(topic)
    print("\n\n===== FINAL REPORT =====\n")
    print(report)