from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# ---- LLM ----
llm = ChatOpenAI(model="gpt-4o-mini")

# ---- Prompts ----
generation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a viral tweet generator."),
    MessagesPlaceholder(variable_name="messages"),
])

reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", "You critique the previous tweet and suggest improvements."),
    MessagesPlaceholder(variable_name="messages"),
])

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm

# ---- Graph ----
graph = StateGraph(MessagesState)

GENERATE = "generate"
REFLECT = "reflect"

def generate_node(state: MessagesState):
    result = generation_chain.invoke({"messages": state["messages"]})
    return {"messages": state["messages"] + [result]}

def reflect_node(state: MessagesState):
    result = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": state["messages"] + [result]}


graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

graph.set_entry_point(GENERATE)

# Loop control
def should_continue(state: MessagesState):
    if len(state["messages"]) > 4:
        return END
    return REFLECT

graph.add_conditional_edges(GENERATE, should_continue)
graph.add_edge(REFLECT, GENERATE)

app = graph.compile()

# ---- Run Agent ----
initial_state = {
    "messages": [
        HumanMessage(content="AI agents taking over content creation")
    ]
}

result = app.invoke(initial_state)

# ---- Print Result ----
print("\nFinal Conversation:\n")
for msg in result["messages"]:
    print(f"{type(msg).__name__}: {msg.content}\n")

# Optional: visualize
mermaid = app.get_graph().draw_mermaid()
print(mermaid)

