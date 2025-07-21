# tests/test_llm_agent_node.py

from langgraph.graph import StateGraph, END
from gemma3n_trial.agents import LLM_Agent  # your class
from gemma3n_trial.schema import DishName
from langchain_google_genai import ChatGoogleGenerativeAI  # or your preferred LLM
import os
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Dummy LangGraph state type
from typing import TypedDict

class DishExtractionState(TypedDict):
    user_query: str
    dish_name: str

# LangGraph node function
def extract_dish_name_node(state: DishExtractionState) -> DishExtractionState:
    agent = LLM_Agent(llm=ChatGoogleGenerativeAI(model="gemma-3n-e4b-it"))  # or your wrapper
    result: DishName = agent.invoke({"user_query": state["user_query"]})
    return {
        "user_query": state["user_query"],
        "dish_name": result.name
    }

# Build LangGraph
def build_graph():
    builder = StateGraph(DishExtractionState)
    builder.add_node("extract_dish_name", extract_dish_name_node)
    builder.set_entry_point("extract_dish_name")
    builder.set_finish_point("extract_dish_name")
    return builder.compile()

# Run the test
if __name__ == "__main__":
    graph = build_graph()
    input_state = {"user_query": "What are ingredients used to make pasta masala?"}
    final_state = graph.invoke(input_state)
    print("Extracted dish name:", final_state["dish_name"])
