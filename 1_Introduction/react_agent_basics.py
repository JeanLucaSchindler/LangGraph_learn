from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.tools import TavilySearchResults
import datetime

load_dotenv()

gemini = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
openai = ChatOpenAI(model="gpt-4.1-mini")

search_tool = TavilySearchResults(search_depth="basic")

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


tools = [search_tool, get_system_time]

react_system_prompt = """
You are a reasoning agent that can use tools.

When solving a problem:
1. Think step-by-step.
2. If you need external information, call the appropriate tool.
3. After receiving tool results, reason about them.
4. Then produce a final clear answer for the user.

Do not expose your internal reasoning unless necessary.
Use tools only when needed.
"""

agent = create_agent(
    tools=tools,
    model=openai,
    system_prompt=react_system_prompt,
)


# Option B — pass message dict like your earlier working example
response = agent.invoke({"messages":[{"role":"user","content":"When was SpaceX's last launch and how many days ago was that from this instant?"}]})
print(response["messages"][-1].content)


