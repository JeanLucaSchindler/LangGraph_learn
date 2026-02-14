from dotenv import load_dotenv
import requests
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

@tool("get_weather", description="Get the current weather for a given location")
def get_weather(location: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Connection": "close"
    }

    response = requests.get(
        f"https://wttr.in/{location}?format=j1",
        headers=headers,
        timeout=(5, 30),
    )

    response.raise_for_status()
    return response.json()

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is the weather like in New York?"}
    ]
})

# print(response["messages"][-1].content)
print(response)
