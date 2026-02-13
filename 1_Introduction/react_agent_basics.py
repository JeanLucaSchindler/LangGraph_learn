from langchain_google_genai import GoogleGenerativeAI
import dotenv
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import TavilySearchResults

dotenv.load_dotenv()

llm = GoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)  

search_tool = TavilySearchResults(search_depth='basic') 
tools = [search_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent.invoke("What is the temperature in Paris right now?")

